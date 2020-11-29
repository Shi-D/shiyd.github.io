[TOC]

# DeepLearning-RNN



## Simple RNN

神经网络训练过程中的两大问题是

* gradient vanishing (梯度消失)
* gradient explode (梯度爆炸)

而对于Simple RNN来说，最大的问题是梯度消失。

解决方法：

GRU、LSTM



## Long Short-Term Memory (LSTM) network

This following figure shows the operations of an LSTM-cell.

<img src="/book/doc/DeepLearning/resources/LSTM.png" alt="4" style="zoom:30%;" />

LSTM-cell above. This tracks and updates a “cell state” or memory variable c^<t>^ at every time-step, which can be different from a^⟨t⟩^. Similar to the RNN example above, you will start by implementing the LSTM cell for a single time-step. Then you can iteratively call it from inside a for-loop to have it process an input with T~x~ time-steps.



### About the gates

<img src="/book/doc/DeepLearning/resources/lstm2.png" alt="4" style="zoom:30%;" />

<img src="/book/doc/DeepLearning/resources/lstm1.png" alt="4" style="zoom:30%;" />



#### - Forget gate

For the sake of this illustration, lets assume we are reading words in a piece of text, and want use an LSTM to keep track of grammatical structures, such as whether the subject is singular or plural. If the subject changes from a singular word to a plural word, we need to find a way to get rid of our previously stored memory value of the singular/plural state. In an LSTM, the forget gate lets us do this:
$$
Γ^{⟨t⟩}_f=σ(W_f[a^{⟨t−1⟩},x^{⟨t⟩}]+b_f)
$$
Here, WfWf are weights that govern the forget gate’s behavior. We concatenate [a^⟨t−1⟩^ , x^⟨t⟩^] and multiply by W~f~. The equation above results in a vector $$Γ^{⟨t⟩}_f$$ with values between 0 and 1. This forget gate vector will be multiplied element-wise by the previous cell state c^⟨t−1⟩^. So if one of the values of  $$Γ^{⟨t⟩}_f$$  is 0 (or close to 0) then it means that the LSTM should remove that piece of information (e.g. the singular subject) in the corresponding component of c^⟨t−1⟩^. If one of the values is 1, then it will keep the information.



#### - Update gate

Once we forget that the subject being discussed is singular, we need to find a way to update it to reflect that the new subject is now plural. Here is the formulat for the update gate:
$$
Γ^{⟨t⟩}_u=σ(W_u[a^{⟨t−1⟩},x^{{t}}]+bu)
$$
Similar to the forget gate, here $$Γ^{⟨t⟩}_u$$ is again a vector of values between 0 and 1. This will be multiplied element-wise with c̃^{⟨t⟩}, in order to compute c^{⟨t⟩}.



#### - Updating the cell

To update the new subject we need to create a new vector of numbers that we can add to our previous cell state. The equation we use is:
$$
\widetilde{c}{⟨t⟩}=tanh⁡(W_c[a^{⟨t−1⟩},x^{⟨t⟩}]+b_c)
$$
Finally, the new cell state is:
$$
\widetilde{c}{⟨t⟩}=tanh(W_c[a^{⟨t−1⟩},x^{⟨t⟩}]+b_c)
$$
Finally, the new cell state is:
$$
c^{⟨t⟩}=Γ^{⟨t⟩}_f∗c^{⟨t−1⟩}+Γ^{⟨t⟩}_u∗\widetilde{c}{⟨t⟩}
$$


#### - Output gate

To decide which outputs we will use, we will use the following two formulas:
$$
Γ^{⟨t⟩}_o=σ(W_o[a^{⟨t−1⟩},x^{⟨t⟩}]+b_o)
$$

$$
a^{⟨t⟩}=Γ^{⟨t⟩}_o∗tanh(c^{⟨t⟩})
$$

Where in equation 6 you decide what to output using a sigmoid function and in equation 7 you multiply that by the tanhtanh of the previous state.



### Forward pass for LSTM

<img src="/book/doc/DeepLearning/resources/forwardpass.png" alt="forwardpass" style="zoom:50%;" />



### Back Propagation

<img src="/book/doc/DeepLearning/resources/backpropogation.png" alt="backpropogation" style="zoom:50%;" />









*我用的展示markdown的框架不能显示内联公式，正在解决中...

*参考资料：https://www.deeplearning.ai/

