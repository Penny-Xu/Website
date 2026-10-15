---
title: "How GPUs Talk to Each Other"
date: "2026-10-14"
tags: ["LLM", "inference", "tensor parallelism", "NCCL", "GPU", "distributed systems"]
description: "A friendly walkthrough of collective communication: what all-reduce is, how ring and tree implementations differ, and why a transformer split across GPUs can't avoid it."
---

A 7B model fits on one GPU. A 70B model does not, so you cut it into pieces and spread the pieces across several. The moment you do that, the GPUs have to talk to each other, because each one is now holding an answer that is the right shape and the wrong value.

That conversation has a name, **all-reduce**. This post is what it is, how it actually gets implemented, and why a transformer can't avoid it.

---

## Part 1: four kids and a pile of candy

Before any of the GPU vocabulary, here is the whole problem with no GPUs in it.

Four kids, Anna, Ben, Cara and Dan, spent recess counting candy scattered around a playground. Each searched a different area, and each counted four colors. So each kid is holding four numbers, and all four of them need the grand total per color for their worksheet.

I'm going to give them deliberately strange counts:

| | RED | BLUE | GREEN | YELLOW |
|---|---|---|---|---|
| **Anna** | 1 | 2 | 3 | 4 |
| **Ben** | 10 | 20 | 30 | 40 |
| **Cara** | 100 | 200 | 300 | 400 |
| **Dan** | 1000 | 2000 | 3000 | 4000 |

Each kid owns a digit place, so any running total announces its own history. `11` means Anna and Ben have contributed. `111` means Anna, Ben and Cara. `1111` means everyone, and it's finished.

One rule: you can say one number to one person at a time, and that takes one second.

### Method 1: everyone tells Anna

The others read their four numbers to Anna, she adds everything, she reads the totals back.

Anna listens to 12 numbers, one at a time, then says 12. **Twenty-four seconds of Anna, while three kids stand there watching.** With twenty kids she listens to 76 and says 76. Adding helpers makes it worse, which is a strange property for teamwork.

### Method 2: pair up, then pair up again

Ben tells Anna his four numbers while Dan simultaneously tells Cara his. Cara tells Anna. Now Anna has everything, and it travels back down the same way.

**Four rounds.** Far better, and it grows slowly: 64 kids needs only 12 rounds, because pairing up halves the group each time. But notice that every single hand-off is all four numbers. Nobody ever passes less than their whole list.

### Method 3: pass to your right

This is the one that took me a while, because the trick is that **four things move at once**.

The kids sit in a circle, each passing only to their right. There isn't one thing being passed around. There are four relay batons, one per color, all travelling simultaneously, each starting from a different seat. A baton carries a running total, and whoever receives it adds their own count for that color.

Follow just the RED baton: starts with Anna carrying `1`, reaches Ben who adds his to make `11`, reaches Cara for `111`, reaches Dan for `1111`. Done. One lap, collecting a number from every kid it visited.

All four are doing that at the same time:

| | RED baton | BLUE baton | GREEN baton | YELLOW baton |
|---|---|---|---|---|
| **start** | Anna `1` | Ben `20` | Cara `300` | Dan `4000` |
| **pass 1** | Ben `11` | Cara `220` | Dan `3300` | Anna `4004` |
| **pass 2** | Cara `111` | Dan `2220` | Anna `3303` | Ben `4044` |
| **pass 3** | Dan **`1111`** | Anna **`2222`** | Ben **`3333`** | Cara **`4444`** |

Read any row across: each kid appears exactly once. Every pass, every kid hands one number right and receives one from the left. Nobody idle, nobody overloaded, and **nobody ever holds more than one number at a time.**

After three passes all four totals are finished but scattered. Dan is the sole expert on red, Anna on blue, Ben on green, Cara on yellow. Three more passes circulate those finished totals around the same circle, with no adding this time, until everyone has all four.

**Six passes. One number each.** Compare Anna in Method 1, who personally handled twenty-four.

The total amount of talking is about the same in all three methods. The ring doesn't do less work. It spreads the work so that nobody is a bottleneck. That is the entire idea, and everything below is bookkeeping about who starts where.

---

## Part 2: the four operations

Those are implementations. The thing being implemented is an **operation**: a contract about who ends up holding what. There are four worth knowing, and they're easy to confuse because three of them involve adding.

Click through this one. It's the same four GPUs throughout, each holding a tensor split into four chunks, and the only thing that changes is where everything ends up.

<iframe src="/blog/tensor-parallelism/collectives.html"
        style="width:100%;height:700px;border:1px solid #d4d4d4;border-radius:4px"
        loading="lazy"></iframe>

---

## Part 3: ring and tree divide different things

Ring is not an alternative to all-reduce. All-reduce is *what*; ring is *how*. Same relationship as sorting and quicksort: identical output, very different cost.

The two that matter in practice are the kids' Method 2 and Method 3, and the cleanest way I've found to hold the difference:

> **Tree divides the participants. Ring divides the data.**

Tree halves the set of GPUs each round, but every message it sends is the full tensor. Ring splits the tensor N ways, but every chunk has to visit every GPU.

| | Rounds | Bytes per message |
|---|---|---|
| **Tree** | `log₂ N` | `D` |
| **Ring** | `N − 1` | `D / N` |

So they win in opposite regimes, and the deciding question is whether you are paying per byte or per round trip:

- **Large messages** → bytes dominate → **ring**. Each GPU moves only `2D·(N−1)/N`, which approaches `2D` and never exceeds it. Doubling the GPU count barely changes what any one GPU sends.
- **Small messages** → round trips dominate → **tree**. The bytes are trivial; what you're paying for is synchronization, and ring has `2(N−1)` of those against tree's `2·log₂ N`. At 64 GPUs that's 126 versus 12.

The good news is that you never pick. **NCCL**, the library sitting underneath `torch.distributed` and therefore underneath vLLM, probes your machine at startup to learn which GPUs share an NVLink and which are stuck behind a PCIe switch, then chooses the algorithm separately for every single call based on how big that message happens to be. The same model will use a tree one millisecond and a ring the next, depending on what it's doing.

If you're curious what it picked, `NCCL_DEBUG=INFO` prints it.

---

## Part 4: why a transformer needs this

Tensor parallelism splits the weight matrices themselves, and [Megatron-LM](https://arxiv.org/abs/1909.08053) showed how to split them so each block costs only one all-reduce instead of two.

The MLP is `Y = GeLU(X·A)`, then `Z = Y·B`. Split `A` by **rows** and each GPU ends up holding a partial sum of `X·A`, which you can't feed to GeLU, because `GeLU(x₁ + x₂) ≠ GeLU(x₁) + GeLU(x₂)`. You'd have to communicate before the nonlinearity, and again after the second matmul.

Split `A` by **columns** instead and each GPU owns a different set of output columns, every value complete. GeLU is elementwise, so it happens locally for free. Then split `B` by **rows** to match, which leaves partial sums, and you reduce once at the end. Attention works the same way: split by head, then split the output projection by rows.

> **Two all-reduces per transformer layer**, one after attention and one after the MLP.

That's a floor, not an implementation detail. It falls out of the algebra, so no amount of engineering removes it. For a 28-layer model, **56 collectives per forward pass.**

---

## Where the time actually goes

Horace He has a piece I'd recommend to anyone, [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html), which argues that before optimizing anything you should ask which of three things is limiting you: **compute**, **memory bandwidth**, or **overhead**. It's the right instinct and it saves enormous amounts of wasted effort.

An all-reduce doesn't fit any of the three. It isn't compute, it isn't memory bandwidth, and it isn't framework dispatch.

His taxonomy was written for single-process PyTorch, where the data is already sitting in memory. Put a process or machine boundary in the path and you need a fourth bucket:

| Bucket | Where the time goes |
|---|---|
| Compute | FLOPs |
| Memory bandwidth | bytes to and from HBM |
| **Transport** | **encode, wire, decode, synchronize across a boundary** |
| Overhead | dispatch, launch, interpreter |

For a model being served across several GPUs, in the regime that actually generates tokens, I suspect the fourth row is where a meaningful share of the bill lands.
