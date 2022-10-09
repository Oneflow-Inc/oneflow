import oneflow
import oneflow.optim as optim
import oneflow.nn as nn
import flowvision

def main():
    model = flowvision.models.resnet.resnet101(pretrained=True).to("cuda")
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=0.9) 

    running_loss = 0.0
    target = oneflow.randn(128, 1000, dtype=oneflow.float32, device="cuda")

    optimizer.zero_grad()
    inputs = oneflow.rand(128, 3, 100, 100, device="cuda" , requires_grad=True)
    outputs = model(inputs)
    loss = criterion(outputs, target)
    loss.backward()
    optimizer.step()
    running_loss += loss.item()

if __name__ == "__main__":
    main()