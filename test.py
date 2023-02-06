# import torch


# # Define a model and an optimizer
# model = YourModel()
# optimizer = torch.optim.Adam(model.parameters())

# # Save the model and the optimizer
# torch.save({
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             }, 'checkpoint.pth')


# import torch

# # Load the saved model and optimizer
# checkpoint = torch.load('checkpoint.pth')
# model = YourModel()
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer = torch.optim.Adam(model.parameters())
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# # Use the loaded model and optimizer
# model.eval()