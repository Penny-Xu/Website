---
title: "Image Restoration"
date: "2019-06-10"
tags: ["image analysis", "restoration filter"]
---
Let's talk about image restoration. It is a fun topic when you understand the intuition.

Say you asked a friend to take a picture of you in front of Lake Geneva. But due to his shaky hands, a blurry bad image was produced.

![](/blog/image-restoration/bad.jpg)

That's quite sad, since you know the image you wanted should have looked like this:

![](/blog/image-restoration/good.jpg)

Now the question is, is it possible to restore the unwanted effects from the bad image, such that we get the good image? Well first, let's understand how we can model the relation between the _ideal_ image $f_{i}$ (good image) and the _observed_ image $f_{o}$ (bad image). Let the mistake and the blur induced by the photographer be modeled by a convolution with a degradation filter and noise. Let $h_{D}$ represent the degradation filter with the assumption that it is a linear low pass filter and $n$ represent the noise with the assumption that it is additive.

$f_{o} = f_{i} \circledast h_{D} + n$

### Inverse Filter

We essentially want to find a filter that will undo the convolution above. We want to find a restoration filter $h_{R}$ that when convolved with the observed image, produces the ideal image. So the restored image $f_{r}$ will be:

$f_{r} = f_{o} \circledast h_{R}$

Let's substitute $f_{o}$ from the model to the above equation:

$f_{r} = (f_{i} \circledast h_{D} + n \ ) \circledast h_{R}$

Let's change the representation to the Fourier domain, so multiplication will take the place of convolution. So the Fourier transform of the restored image will be the Fourier transform of the ideal image multiplied with the Fourier transform of the degradation filter plus noise, all multiplied by the Fourier transform of the restoration filter:

$F_{r} = (F_{i} \cdot H_{D} + N \ ) \cdot H_{R}$

Now do you see what might be a good idea for the restoration filter $H_{R}$? It is exactly the __inverse filter__. Do you see why it is called the inverse filter?

$H_{R} = {1 \over H_{D}}$

This means that if we know $H_{D}$ or an approximation of it, we can build an inverse filter to restore our observed image. Thus with substitution of $H_{R}$, the spectrum of the restored image will be:

$F_{r} = F_{i}  + {N  \over H_{D}}$

Well this is great since the equation above shows that the restored image is really just the ideal image plus some high pass noise and that the restored image approaches the ideal image when noise is low. This means that the inverse filter can be a very good solution to our problem!

However, what if the noise is large? Then wouldn't ${N  \over H_{D}}$ increase and corrupt our image?...

### Adaptive Filter

When using the inverse filter on images where there is no noise, the image restoration would be perfect. However, the problem is that when there is noise, the noise will be amplified. This is why will look at the __Weiner filter__, which is a type of adaptive filter. The difference between the inverse and the Weiner filters is that the inverse filter disregard the noise, while the Weiner filter considers a model of both the image and the noise.

The Weiner filter aims to find a $H_R$ such that the error between the restored image and the ideal image is minimal:

$\varepsilon = E\{(f_i - f_r)^2\}$

In order to solve for $H_R$, we can take the derivative of the error with respect to $H_R$ and set it to zero. Remember, $H_R$ is in the term $f_r$ in the error equation since $f_r$ is $f_o$ convolved with $H_R$. Without going into the derivation of $H_R$ as it simply involves taking derivatives, below is the solution $H_R$, the Weiner filter, in the Fourier domain:

$H_R = {P_{f_i f_o} \over P_{f_o}}$

where $P_{f_i f_o}$ is the power interspectrum, which is related to the complex conjugate of the degradation filter and the power spectrum of the ideal image. $P_{f_o}$ is the power spectrum of $f_o$, which is related to the power spectrum of the noise and the power spectrum of the ideal image. When the noise is additive, we can write $P_{f_i f_o}$ and $P_{f_o}$ as:

$P_{f_i f_o} = H_{D}^* P_{f_i}$

$P_{f_o} = |H_{D}|^2 P_{f_i} + P_N$

Finally, $H_R$ is as follows:

$H_R = {H_{D}^* P_{f_i} \over |H_{D}|^2 P_{f_i} + P_N}$

We can see from the equation above that the Wiener filter has two separate part, an inverse filtering part and a noise smoothing part. It not only performs the deconvolution by inverse filtering (highpass filtering) but also removes the noise with a compression operation (lowpass filtering).

More intuitively, the Weiner filter does this: for each frequency, we can estimate if the power of the noise is significant in respect to the power of the image in order to see whether it is a good idea to use the inverse filter or not. This means that adaptively, depending on the content of the noise and the content of the image, each frequency will be amplified by a value ranging from zero, when there is a lot of noise, to the inverse filter, when there is no noise. The values in between, it finds the best tradeoff to do a certain degree of the inverse filter. Essentially, the Weiner filter behaves like the inverse filter at low frequencies and like a low-pass filter for high frequencies.

I hope now you understand the main idea of image restoration through inverse filters and Weiner filters. Have a great day! :)
