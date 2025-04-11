# A logistic regression forward pass

import torch.nn.functional as F  # A
import torch
from torch.autograd import grad

y = torch.tensor([1.0])  # B
x1 = torch.tensor([1.1])  # C
w1 = torch.tensor([2.2], requires_grad=True)  # D
b = torch.tensor([0.0], requires_grad=True)  # E
z = x1 * w1 + b  # F
a = torch.sigmoid(z)  # G

loss = F.binary_cross_entropy(a, y)

# A 这是 PyTorch 中常见的导入约定，用于避免代码行过长
# B 真实标签
# C 输入特征
# D 权重参数
# E 偏置单元
# F 网络输入
# G 激活与输出

grad_L_w1 = grad(loss, w1, retain_graph=True)  # A
grad_L_b = grad(loss, b, retain_graph=True)
print(grad_L_w1)
print(grad_L_b)

# 自动反向传播计算loss
loss.backward()
print(w1.grad)
print(b.grad)


# A multilayer perceptron with two hidden layers

class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):  # A
        super().__init__()
        self.layers = torch.nn.Sequential(
            # 1st hidden layer
            torch.nn.Linear(num_inputs, 30),  # B
            torch.nn.ReLU(),  # C

            # 2nd hidden layer
            torch.nn.Linear(30, 20),  # D
            torch.nn.ReLU(),

            # output layer
            torch.nn.Linear(20, num_outputs),
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits  # E


# A 将输入和输出的数量编码为变量很有用，这样可以为具有不同特征和类别数量的数据集重用相同的代码。
# B Linear 层将输入和输出节点的数量作为参数。
# C 非线性激活函数放置在隐藏层之间。
# D 一个隐藏层的输出节点数必须与下一个隐藏层的输入节点数相匹配。
# E 最后一层的输出被称为 logits。

model = NeuralNetwork(50, 3)
print(model)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of trainable model parameters:", num_params)

torch.manual_seed(123)
model = NeuralNetwork(50, 3)
print(model.layers[0].weight)

# 前向传播
torch.manual_seed(123)
X = torch.rand((1, 50))
out = model(X)
print(out)

# 当我们将模型用于推理（例如，进行预测）而不是训练时，最佳实践是使用 torch.no_grad() 上下文管理器，这告诉pytorch不需要训练跟踪梯度
with torch.no_grad():
    out = torch.softmax(model(X), dim=1)
print(out)

import torch.nn.functional as F

torch.manual_seed(123)
model = NeuralNetwork(num_inputs=2, num_outputs=2)  # A
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)  # B

'''
数据加载器
'''
X_train = torch.tensor([
    [-1.2, 3.1],
    [-0.9, 2.9],
    [-0.5, 2.6],
    [2.3, -1.1],
    [2.7, -1.5]
])
y_train = torch.tensor([0, 0, 0, 1, 1])

X_test = torch.tensor([
    [-0.8, 2.8],
    [2.6, -1.6],
])
y_test = torch.tensor([0, 1])

from torch.utils.data import Dataset


class ToyDataset(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y

    def __getitem__(self, index):  # A
        one_x = self.features[index]  # A
        one_y = self.labels[index]  # A
        return one_x, one_y  # A

    def __len__(self):
        return self.labels.shape[0]  # B


train_ds = ToyDataset(X_train, y_train)
test_ds = ToyDataset(X_test, y_test)

# Listing A.7 Instantiating data loaders

from torch.utils.data import DataLoader

torch.manual_seed(123)

train_loader = DataLoader(
    dataset=train_ds,       #A
    batch_size=2,
    shuffle=True,           #B
    num_workers=0,           #C
    drop_last=True
)

test_loader = DataLoader(
    dataset=test_ds,
    batch_size=2,
    shuffle=False,          #D
    num_workers=0
)


#A 之前创建的 ToyDataset 实例作为数据加载器的输入。
#B 是否打乱数据
#C 后台进程的数量
#D 没有必要打乱测试数据

'''
批量训练
'''
num_epochs = 3

for epoch in range(num_epochs):

    model.train()
    for batch_idx, (features, labels) in enumerate(train_loader):
        logits = model(features)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()  # C
        loss.backward()  # D
        optimizer.step()  # E

        ### LOGGING
        print(f"Epoch: {epoch + 1:03d}/{num_epochs:03d}"
              f" | Batch {batch_idx:03d}/{len(train_loader):03d}"
              f" | Train Loss: {loss:.2f}")

    model.eval()
    # Optional model evaluation

# A 上一节的数据集包含 2 个特征和 2 个类别
# B 我们让优化器知道需要优化哪些参数
# C 将上一轮的梯度设置为零，以防止意外的梯度累积
# D 计算损失函数相对于模型参数的梯度
# E 优化器使用梯度来更新模型参数


# 训练完再进行预测
model.eval()
with torch.no_grad():
    outputs = model(X_train)
print(outputs)

torch.set_printoptions(sci_mode=False)
probas = torch.softmax(outputs, dim=1)
print(probas)

