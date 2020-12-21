We implement SGD with Momentum or Nesterov Accelerated Gradient, and Adam to estimate the parameters of a logistic regression model, and 
empirically compare their performance. 

Consider a dataset with $$N$$ samples, <a href="https://www.codecogs.com/eqnedit.php?latex=D&space;=&space;\{(x^{(i)},&space;y^{(i)})\}^N_{i&space;=&space;1},&space;\text{where}&space;\;&space;x^{(i)}&space;\in&space;\mathbb{R}^2&space;\;&space;\text{and}&space;\;&space;y^{(i)}&space;\in&space;\{-1,&space;&plus;1\}.&space;\text{We&space;will&space;fit&space;a&space;logistic&space;regression&space;model}:\\&space;p(y^{(i)}|&space;x^{(i)})&space;=&space;\sigma(y^{(i)}&space;w^T&space;x^{(i)})," target="_blank"><img src="https://latex.codecogs.com/gif.latex?D&space;=&space;\{(x^{(i)},&space;y^{(i)})\}^N_{i&space;=&space;1},&space;\text{where}&space;\;&space;x^{(i)}&space;\in&space;\mathbb{R}^2&space;\;&space;\text{and}&space;\;&space;y^{(i)}&space;\in&space;\{-1,&space;&plus;1\}.&space;\text{We&space;will&space;fit&space;a&space;logistic&space;regression&space;model}:\\&space;p(y^{(i)}|&space;x^{(i)})&space;=&space;\sigma(y^{(i)}&space;w^T&space;x^{(i)})," title="D = \{(x^{(i)}, y^{(i)})\}^N_{i = 1}, \text{where} \; x^{(i)} \in \mathbb{R}^2 \; \text{and} \; y^{(i)} \in \{-1, +1\}. \text{We will fit a logistic regression model}:\\ p(y^{(i)}| x^{(i)}) = \sigma(y^{(i)} w^T x^{(i)})," /></a>


<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mi>p</mi>
  <mo stretchy="false">(</mo>
  <msup>
    <mi>y</mi>
    <mrow>
      <mo stretchy="false">(</mo>
      <mi>i</mi>
      <mo stretchy="false">)</mo>
    </mrow>
  </msup>
  <mo stretchy="false">|</mo>
  <msup>
    <mi>x</mi>
    <mrow>
      <mo stretchy="false">(</mo>
      <mi>i</mi>
      <mo stretchy="false">)</mo>
    </mrow>
  </msup>
  <mo stretchy="false">)</mo>
  <mo>=</mo>
  <mi>&#x03C3;<!-- σ --></mi>
  <mo stretchy="false">(</mo>
  <msup>
    <mi>y</mi>
    <mrow>
      <mo stretchy="false">(</mo>
      <mi>i</mi>
      <mo stretchy="false">)</mo>
    </mrow>
  </msup>
  <msup>
    <mi>w</mi>
    <mi>T</mi>
  </msup>
  <msup>
    <mi>x</mi>
    <mrow>
      <mo stretchy="false">(</mo>
      <mi>i</mi>
      <mo stretchy="false">)</mo>
    </mrow>
  </msup>
  <mo stretchy="false">)</mo>
  <mo>,</mo>
</math>






The problem can then be solved by minimizing the negative log likelihood. Recall (from class) that this results in the logistic loss function :
L(w) = \frac{1}{N} \sum_{(x^{(i)}, y^{(i)}) \in D} \log(1 + \exp(-y^{(i)} w^Tx^{(i)})).
We are interested in estimating w^* such that:
w^* = \underset{w}{\operatorname{argmin}}L(\{(x^{(i)}, y^{(i)})\}_0^{n-1}; w).
As considered in class, we will compute the gradient of the loss on subsamples (batches) of data.

NOTE: For this assignment we will be using a batch size of 1, i.e., computing updates using individual points. You should compute the gradient and update the weights for each point in the dataset for every iteration. Recall that SGD is described by the following update step:
w^{(t+1)} \leftarrow w^{(t)} - \alpha\nabla_w L(\{(x^{(t)}, y^{(t)})\}; w^{(t)})
Notice that only 1 point is passed into the loss function.
Nesterov Momentum

While basic SGD is the simplest iterative update, we can often achieve faster convergence by including Nesterov Momentum. From a physical perspective, the loss can be interpreted as the height of a hilly terrain and the optimization process with momentum term can then be seen as equivalent to the process of simulating a ball rolling down the hill.

The resulting algorithm is described by the following update equations:
\begin{align}
    \hat{w}^{(t-1)} \leftarrow& w^{(t-1)}-\beta V^{(t-1)} \\
    V^{(t)} \leftarrow& \beta V^{(t-1)} + \alpha \nabla_wL(\{(x^{(t)}, y^{(t)})\};\hat{w}^{(t-1)})\\
    w^{(t)} \leftarrow& w^{(t-1)} - V^{(t)}
    \end{align}
Where \beta is called the momentum term. The name momentum stems from an analogy to momentum in physics: the weight vector w, thought of as a particle traveling through parameter space, incurs acceleration from the gradient of the loss ("force"). Unlike in classical stochastic gradient descent, it tends to keep traveling in the same direction, preventing oscillations. (cite from Wikipedia)
Adam

Adaptive Moment Estimation (Adam) is another optimization method. It is an an update to the RMSProp optimizer. In this optimization algorithm, running averages of both the gradients and the second moments of the gradients are used. The Adam algorithm is described by the following update equations:
\begin{align}
    m^{(t)} \leftarrow& \beta_1 m^{(t-1)} + (1 - \beta_1) \nabla_w L(\{(x^{(t)}, y^{(t)})\}; w^{(t-1)}) \\
    v^{(t)} \leftarrow& \beta_2 v^{(t-1)} + (1 - \beta_2) \nabla_w L(\{(x^{(t)}, y^{(t)})\}; w^{(t-1)})^2 \\
    \hat{m} \leftarrow& \frac{m^{(t)}}{1 - \beta_1} \\
    \hat{v} \leftarrow& \frac{v^{(t)}}{1 - \beta_2} \\
    w^{(t)} \leftarrow& w^{(t-1)} - \frac{\alpha}{\sqrt{\hat{v}} + \epsilon} \hat{m}
    \end{align}
where \epsilon is a small scalar used to prevent division by 0, and \beta_{1} and \beta_{2} are the forgetting factors for gradients and second moments of gradients, respectively. Squaring and square-rooting is done elementwise. (Wikipedia)
Implement:

You are going to implement five functions:

    gradient_loss(y, x, w) This will be calculated over a single point
    loss_fn(y, x, w) This should be implemented to account for the entire dataset or a single point.
    sgd(pnts, nepochs, alpha, w0)
    sgd_nesterov(pnts, nepochs, alpha, w0, momentum)
    adam(pnts, nepochs, alpha, w0, betas, epsilon)

We provide the data in the following variable:
data_points is an N-by-4 numpy array with the following structure:
\begin{bmatrix}
x_1^{(1)} & x_2^{(1)} & 1 & y^{(1)} \\
x_1^{(2)} & x_2^{(2)} & 1 & y^{(2)} \\
\vdots & \vdots & \vdots & \vdots \\
x_1^{(N)} & x_2^{(N)} & 1 & y^{(N)} \\
\end{bmatrix}
All function calls are already implemented for you, just write the five functions listed above.
