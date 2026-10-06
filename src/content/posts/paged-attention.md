---
title: "PagedAttention, Visualized"
date: "2026-10-05"
tags: ["LLM", "inference", "vLLM", "KV cache", "GPU", "operating systems"]
---

I spent the last week trying to understand how large language models are actually served, not how they are trained. Somewhere in the middle of that I rented a GPU, ran my own benchmarks, and read the [PagedAttention paper](https://arxiv.org/abs/2309.06180). About four pages in I realized the whole thing is an operating systems paper wearing a machine learning hat.

The central claim is this: when you serve an LLM, the thing that limits you is not how fast the GPU can multiply matrices. It is memory. And the fix the authors propose is virtual memory paging, applied to a place nobody had applied it before.

This post is my attempt to see that clearly. Three visualizations, each one a step-through you click at your own pace. Like my other posts, this is not meant to teach you how to implement anything. It is meant to help you _see_ what is going on.

---

## Part 1: what one token actually costs

Before memory makes sense, the forward pass has to make sense. So let's follow the prompt "hey my friend" all the way through a model until it produces a single next token.

Qwen2.5-7B is the model I used throughout, so the numbers below are its: 28 layers, hidden dimension 3584, vocabulary of about 152,000.

<iframe src="/blog/paged-attention/forward-pass.html"
        style="width:100%;height:660px;border:1px solid #d4d4d4;border-radius:4px"
        loading="lazy"></iframe>

The part I want you to notice is the shape. The matrix goes in as `[3 × 3584]` and comes out of all 28 layers as `[3 × 3584]`. Nothing is reshaped along the way. Each layer just refines the values, with attention letting tokens look at each other and the MLP processing each token alone.

And at the end, after all of that work, you get **one token**. To get the next one, the entire pass runs again.

---

## Part 2: what the KV cache actually holds

Running the whole model again for every single token sounds absurd, and it would be, if you also had to recompute everything about the tokens that came before. You don't. That is what the KV cache is for.

The part that took me a while to internalize: **Q is thrown away, K and V are kept.** A token's query gets used once, to compute that token's own output. But its key and value are read by every token that comes after it.

<iframe src="/blog/paged-attention/kv-cache.html"
        style="width:100%;height:640px;border:1px solid #d4d4d4;border-radius:4px"
        loading="lazy"></iframe>

Which gives you the number that everything else in this post depends on:

```
2 (K and V) × 512 × 28 layers × 2 bytes (FP16) = 57,344 bytes
```

About **56 KB per token**. Per user. Held in GPU memory for as long as that request is alive.

I did not trust this until I checked it against something real. When I served the model on an A40, vLLM reported 23.88 GiB of KV cache holding 447,168 tokens. Divide those:

```
23.88 × 1024³ ÷ 447,168 ≈ 57,335 bytes per token
```

Which is the arithmetic above, to within rounding. That was the moment the whole thing stopped being abstract for me.

Scale it up and the problem is obvious. A 32,000-token conversation is about **1.8 GB of KV cache for one user**. On a 46 GB card, you can hold a startlingly small number of long conversations at once. And since serving throughput depends almost entirely on how many requests you can batch together, and batching depends on how many KV caches fit in memory, **memory capacity is throughput**.

---

## Part 3: it is just paging

So you need to hold as many conversations in memory as possible. The problem is that you have no idea how long any of them will be. A response might be 20 tokens or 2,000, and you have to allocate memory before you find out.

The old answer was to reserve space for the worst case. If the model supports 32,768 tokens of context, reserve 32,768 tokens of KV cache per request, contiguously. A conversation that ends up using 500 tokens wastes about 98% of what it was given.

If you have ever taken an operating systems course, you already know the fix, because this is textbook internal fragmentation and the answer is textbook too. Stop handing out contiguous ranges. Chop memory into small fixed-size blocks, hand them out one at a time as a sequence grows, and keep a per-request table mapping the sequence's logical positions to wherever its blocks physically landed.

That table is a page table. The blocks are pages. The whole thing is paging.

<iframe src="/blog/paged-attention/paged-memory.html"
        style="width:100%;height:700px;border:1px solid #d4d4d4;border-radius:4px"
        loading="lazy"></iframe>

The mapping is almost suspiciously exact:

| Operating system | PagedAttention |
|---|---|
| Virtual address space | Logical token positions |
| Physical memory | GPU KV cache memory |
| Page | Block (16 tokens) |
| Page table | Block table |
| Internal fragmentation | Wasted reserved KV space |
| Copy-on-write after `fork()` | Shared prefix blocks |
| Swapping to disk | Evicting blocks to CPU memory |

The copy-on-write one is my favorite, because of how much it buys. Every request to a chatbot carries the same system prompt. Identical tokens produce identical keys and values, so every one of those requests can point at the _same physical blocks_ for the shared part, with a reference count tracking how many are using it. A thousand users, one copy of the prompt. And when a sequence gets evicted under memory pressure, its shared blocks don't even have to move, because somebody else still needs them.

---

## What I actually measured

Reading the paper is one thing. I wanted numbers, so I rented an A40 and benchmarked Qwen2.5-7B two ways: full precision, and 4-bit quantized with AWQ. Same model, same workload, same fixed random seed, back to back.

| | FP16 | AWQ | |
|---|---|---|---|
| Model weights | 14.29 GiB | 5.29 GiB | −63% |
| KV cache available | 23.88 GiB | 33.82 GiB | +42% |
| Concurrent request ceiling | 13.65× | 19.32× | +42% |
| Throughput @ concurrency 1 | 34 tok/s | 105 tok/s | 3.1× |
| Throughput @ concurrency 32 | 868 tok/s | 1,879 tok/s | 2.2× |

Two things in there surprised me.

**The 9 GiB freed from weights turned into 9.94 GiB of KV cache, almost one for one.** Weights and cache compete for the same fixed pool, so shrinking one directly expands the other. Quantization is not only a speed optimization, it is a capacity one. The concurrency ceiling rose 42% on identical hardware.

**And the speedup shrinks as you add load** — 3.1× at concurrency 1, down to 2.2× at concurrency 32. That one makes sense once you think about where the time goes. At batch size 1 the GPU reads every weight in the model to produce a single token, so it is almost entirely waiting on memory, and halving the bytes nearly halves the time. As the batch fills up, that same weight read gets amortized across many requests at once, compute starts to matter, and the advantage of having fewer bytes narrows.

Which is the paper's thesis, arrived at from the other direction. Compute is abundant. Memory is not. Nearly everything interesting in LLM serving right now is a memory management problem, and operating systems have been solving those for fifty years.
