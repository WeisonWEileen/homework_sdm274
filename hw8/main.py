import hydra
from omegaconf import DictConfig
from tensor.Tensor_softmax import *
from utils.utils import *

@hydra.main(version_base="1.3", config_path="./conf", config_name="config")
def main(cfg: DictConfig):
    
    train_set_path, test_set_path, units, batch_size, epoches, lr = get_hyperparam(cfg)

    train_data, train_labels = read_mnist_data(train_set_path)
    test_data, test_labels = read_mnist_data(test_set_path)
    
    # linear_layer_1 = Linear((units[0], units[1]), batch_size, "LinearLayer1")
    linear_layer_1 = Linear((64,10), batch_size, "LinearLayer1")
    # linear_layer_1 = Linear((units[0], units[1]), batch_size, "LinearLayer1")
    # linear_layer_2 = Linear((units[1], units[2]), batch_size, "LinearLayer2")
    losses = []

    for epoch in range(epoches):
        random_indexes = np.random.choice(train_data.shape[0], batch_size, replace=False)
        
        inputs = Tensor(train_data[random_indexes], name="inputs")
        targets = Tensor(train_labels[random_indexes], name="targets")

        z1 = linear_layer_1.forward(inputs)
        # a1 = z1.relu()
        # z2 = linear_layer_2.forward(a1)

        a2 = z1.softmax()
        loss = a2.cross_entropy_loss(targets)
        losses.append(loss.data)
        loss.backward()

        linear_layer_1.weights.data -= 1 / batch_size * lr * linear_layer_1.weights.grad
        linear_layer_1.bias.data -= 1 / batch_size * lr * np.sum(linear_layer_1.bias.grad, axis=0, keepdims=True)
        loss.zero_grad()

        # linear_layer_2.weights.data -= 1 / batch_size * lr * linear_layer_2.weights.grad
        # linear_layer_2.bias.data -= 1 / batch_size * lr * np.sum(linear_layer_2.bias.grad, axis=0, keepdims=True)



        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss.data:.4f}")
    
    # plot
    plot_loss_curve(losses, "Loss curve")

    test_inputs = Tensor(test_data, name="test_inputs")


    z = linear_layer_1.forward(test_inputs)
    a = z.softmax()
    precision = one_hot_precision(a.data, test_labels)
    print(precision)

    # a = 
        


if __name__ == "__main__":
    main()
