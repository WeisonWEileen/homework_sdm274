import hydra
from omegaconf import DictConfig
from tensor.Tensor_softmax import *
from utils.utils import *

@hydra.main(version_base="1.3", config_path="./conf", config_name="config")
def main(cfg: DictConfig):
    
    train_set_path, test_set_path, units, batch_size, epoches, lr = get_hyperparam(cfg)

    train_data, train_labels = read_mnist_data(train_set_path)
    test_data, test_labels = read_mnist_data(test_set_path)
    
    linear_layer_1 = Linear((units[0], units[1]), batch_size, "LinearLayer1")
    linear_layer_2 = Linear((units[1], units[2]), batch_size, "LinearLayer2")



    for epoch in range(epoches):
        random_indexes = np.random.choice(train_data.shape[0], batch_size, replace=False)
        # print(random_indexes)
        
        inputs = Tensor(train_data[random_indexes], name="inputs")
        targets = Tensor(train_labels[random_indexes], name="targets")

        z1 = linear_layer_1.forward(inputs)
        a1 = Tensor.relu(z1)
        z2 = linear_layer_2.forward(a1)
        a2 = Tensor.softmax(z2)
        loss = a2.cross_entropy_loss(targets)
        loss.backward()

        linear_layer_1.weights.data -= 1 / batch_size * lr * linear_layer_1.weights.grad
        linear_layer_1.bias.data -= 1 / batch_size * lr * np.sum(linear_layer_1.bias.grad, axis=0, keepdims=True)
        linear_layer_2.weights.data -= 1 / batch_size * lr * linear_layer_2.weights.grad
        linear_layer_2.bias.data -= 1 / batch_size * lr * np.sum(linear_layer_2.bias.grad, axis=0, keepdims=True)

        # print(linear_layer_1.weights.grad)
        # print(linear_layer_1.bias.grad)
        # print(linear_layer_2.weights.grad)
        # print(linear_layer_2.bias.grad)

        loss.zero_grad()
        # linear_layer_1.weights.zero_grad()
        # linear_layer_1.bias.zero_grad()
        # linear_layer_2.weights.zero_grad()
        # linear_layer_2.bias.zero_grad()

        # if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.data:.4f}")
        


if __name__ == "__main__":
    main()
