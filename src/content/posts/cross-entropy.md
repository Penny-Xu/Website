---
title: "Do you even know cross entropy?"
date: "2019-06-01"
tags: ["cross entropy", "entropy", "machine learning"]
---
Ok so this is a bit embarrassing. A friend asked me this question when explaining some concepts in this Big Data class I was taking at the time. Although I have learned entropy in my physics, computer engineering, and probability classes, I was unable to explain what it was exactly. Honestly, the thought of entropy, cross entropy, and KL divergence used to scare me...it just sounds so technical. However, engineers just love to make easy things seem difficult!

## Entropy

Ok so we are going to talk about this "entropy" term in respect to bits. Say there are 3 people (X, Y, and Z) who share a phone messaging plan. Say person X is from generation X and doesn't really use messaging, so he uses around 20% of the plan. Say person Y is from generation Y (millennials), so he will use around 30% of the plan. Finally, person Z is from generation Z, and of course he uses 50% of the plan. Let's call this distribution y:

    distribution y:   X = 0.2
                      Y = 0.3
                      Z = 0.5

Now, every time a person in this plan sends a message, the sender is identified through an unique bit sequence. The longer the bit sequence, the more costly it is per message. Let's randomly assign a bit encoding described below:

    encoding #1:   X   --> (0)  1 bit
                   Y   --> (10) 2 bits
                   Z   --> (11) 2 bits

We can calculate the average number of bits used to send messages in this phone plan based on the encoding #1:

(0.2 $\cdot$ 1 bit) + (0.3 $\cdot$ 2 bits ) + (0.5 $\cdot$ 2 bits) = __1.8 bits__

But can we do better than 1.8 bits? What if we encode the sender knowing the underlying distribution, for example, Z sends a lot of messages, so let's swap his encoding with X. Let's look at another encoding.

    encoding #2:   X   --> (11)  2 bits
                   Y   --> (10)  2 bits
                   Z   --> (0)   1 bit

We can calculate the average number of bits used to send messages in this phone plan based on the encoding #2:

(0.2 $\cdot$ 2 bits) + (0.3 $\cdot$ 2 bits ) + (0.5 $\cdot$ 1 bit) = __1.5 bits__

We can see that encoding #2 results in a smaller average number of bits. Now the question is, how do we know if our encoding is the best one that achieves the smallest average encoding size? Now the answer to this question is exactly "entropy."

$entropy = \sum\nolimits_{} y_i log {1\over y_i}$


We can always find entropy given a distribution. In the equation above, $log {1\over y_i}$ represents the number of bits to assign to the $i^{th}$ symbol, in our case, person X, Y, and Z. This makes sense intuitively because if $y_i$ is large, then $log {1\over y_i}$ would be small. This means that we want to use less bits to encode a more frequent symbol, in our case, we want to use the lest amount of bits for person Z.

Let's calculate the entropy for the above distribution y:

$$ entropy(y) = \sum\nolimits_{} y_i log {1\over y_i} $$

$$ entropy(y) = y_x \cdot log {1\over y_x} + y_y \cdot log {1\over y_y} + y_z \cdot log {1\over y_z} $$

$$ entropy(y) = 0.2 \cdot log {1\over 0.2} + 0.3 \cdot log {1\over 0.3} + 0.5 \cdot log {1\over 0.5} $$

$$ entropy(y) = 0.2 \cdot 2.32 + 0.3 \cdot 1.74 + 0.5 \cdot 1 $$

$$ entropy(y) = 1.486 $$

## Cross Entropy

For entropy, we effectively used our knowledge of distribution y to assign the bits for each person so that the total cost would be the minimum. But what if we didn't know the _correct_ distribution? What if we thought that it was actually distribution x? Then cross entropy calculates the average number of bits used using the _wrong_ distribution.

$crossEntropy(y, x) = \sum\nolimits_{} y_i log {1\over x_i}$

You see that the only difference between the entropy equation and the cross entropy equation is $log {1\over x_i}$. This makes sense since now we are encoding the  $i^{th}$ symbol using  $log {1\over x_i}$ bits instead of $log {1\over y_i}$ bits.

    distribution x:   X = 0.2
                      Y = 0.5
                      Z = 0.3


$$ crossEntropy(y, x) = \sum\nolimits_{} y_i log {1\over x_i} $$

$$ crossEntropy(y, x) = y_x \cdot log {1\over x_x} + y_y \cdot log {1\over x_y} + y_z \cdot log {1\over x_z} $$

$$ crossEntropy(y, x) = 0.2 \cdot log {1\over 0.2} + 0.3 \cdot log {1\over 0.5} + 0.5 \cdot log {1\over 0.3} $$

$$ crossEntropy(y, x) = 0.2 \cdot 2.32 + 0.3 \cdot 1 + 0.5 \cdot 1.74 $$

$$ crossEntropy(y, x) = 1.634 $$



## KL Divergence

Stands for _Kullback-Leibler divergence_, which is literally just the difference between cross entropy and entropy. Remember, cross entropy is always greater or equal to entropy and it's shown in the example above. Basically, KL divergence means how much _extra_ bits is used if distribution x is used, rather than distribution y. In a more general sense, it is a measure of how a probability distribution differs from another probability distribution.

$$D_{KL}(y, x) = crossEntropy(y,x) - entropy(y) = 1.634 - 1.486$$

## Conclusion

I was asked by a kind viewer: why should we understand entropy/cross entropy? And what are some practical applications? Well, below is my response:

The "entropy" described in the post is really entropy relating to information theory, thanks to Claude Shannon. A practical situation would be the original problem Claude Shannon was trying to solve, which is: what is the most efficient way to send messages without losing information? The efficiency described here would be the average message length. So given a distribution with random variables (X, Y, Z, ...), the least amount of bits to encode symbol X (a type of message) would be log(1/probability(X)). Then a good feature to summarize/quantify this information would be "entropy," which is the smallest possible average size of lossless encoding of the messages. But really, I think about "entropy" as a characteristic of a distribution. It tells me something about the distribution, for example, if the entropy of a message distribution is high (the average number of bits is large), then this entropy tells me that there is uncertainty or unpredictability of the next message. Inversely, low entropy tells me that the next message is predicable, so there is less disorder.

But more importantly, I think understanding entropy this way helps me understand cross entropy, which is very important in creating and training models for machine learning. For example, if a machine learning model outputs a distribution x, but the true distribution is distribution y, then cross entropy tells me how wrong the model is. Therefore, cross entropy is often used as a loss model, and minimizing the cross entropy will help the model "learn."
