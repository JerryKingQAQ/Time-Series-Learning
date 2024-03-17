import numpy as np
import torch


def train(
    model, train_loader, test_loader, loss_fn, optimizer, scheduler, n_epochs=200
):
    best_score = None
    best_weights = None
    best_train_preds = None
    best_test_preds = None

    for epoch in range(n_epochs):
        test_preds = []
        train_preds = []

        model.train()
        train_squared_errors_sum = 0.0
        train_sample_count = 0
        for inputs, targets in train_loader:
            predictions = model(inputs)
            loss = loss_fn(predictions, targets)  # 确保loss_fn可以处理这种形状
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_preds.append(predictions.detach().cpu().numpy())

            # 累加平方误差和样本数量
            train_squared_errors_sum += ((predictions - targets) ** 2).sum().item()
            train_sample_count += (
                targets.numel()
            )  # 使用numel()确保考虑到整个预测范围内的所有元素

        train_rmse = np.sqrt(train_squared_errors_sum / train_sample_count)

        model.eval()
        test_squared_errors_sum = 0.0
        test_sample_count = 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                predictions = model(inputs)
                # 累加平方误差和样本数量
                test_squared_errors_sum += ((predictions - targets) ** 2).sum().item()
                test_sample_count += targets.numel()
                test_preds.append(predictions.detach().cpu().numpy())

        test_rmse = np.sqrt(test_squared_errors_sum / test_sample_count)

        # 更新学习率调度器
        scheduler.step(test_rmse)

        if best_score is None or test_rmse < best_score:
            best_score = test_rmse
            best_weights = model.state_dict()
            best_train_preds = train_preds
            best_test_preds = test_preds

        # 每隔一定epoch输出一次信息
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Train RMSE: {train_rmse}, Test RMSE: {test_rmse}")

    # model.load_state_dict(best_weights)
    return np.array(best_train_preds, dtype=object), np.array(
        best_test_preds, dtype=object
    )
