import metatomic.torch as mta
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch import nn

from src.mad_explorer.explorer import MADExplorer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1k PET-MAD last-layer features selected by FPS
features_tensor = torch.load("mad_features.pt", map_location=device)

# Corresponding sketchmap projections
targets_tensor = torch.load("smap_3d_projection.pt", map_location=device)


model = MADExplorer("pet-mad-latest.ckpt", device=device)

model.feature_scaler.fit(features_tensor)
model.projection_scaler.fit(targets_tensor)

scaled_features = model.feature_scaler.transform(features_tensor)
scaled_targets = model.projection_scaler.transform(targets_tensor)


X_train, X_test, y_train, y_test = train_test_split(
    scaled_features.cpu().numpy(),
    scaled_targets.cpu().numpy(),
    test_size=0.2,
    random_state=42,
)

X_train = torch.FloatTensor(X_train).to(device)
y_train = torch.FloatTensor(y_train).to(device)
X_test = torch.FloatTensor(X_test).to(device)
y_test = torch.FloatTensor(y_test).to(device)


criterion = nn.SmoothL1Loss(reduction="mean")
optimizer = optim.Adam(model.projector.parameters(), lr=0.0005, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, "min", patience=5, factor=0.5
)

num_epochs = 100
best_val_loss = float("inf")
patience = 10
counter = 0

save_checkpoint_name = "mad_explorer_checkpoint.ckpt"

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    outputs = model.projector(X_train)
    loss = criterion(outputs, y_train)

    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_outputs = model.projector(X_test)
        val_loss = criterion(val_outputs, y_test)

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch+1}, train loss: {loss.item():.4f}, val loss {val_loss.item():.4f}"
        )

    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0

        model.save_checkpoint(save_checkpoint_name)
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping")
            break


# Creation of metatomic model

metadata = mta.ModelMetadata(
    name="mad-explorer",
    description="Exploration tool for PET-MAD model features upon SMAP projections",
)

outputs = {"features": mta.ModelOutput(per_atom=True)}

capabilities = mta.ModelCapabilities(
    outputs=outputs,
    length_unit="angstrom",
    supported_devices=["cpu", "cuda"],
    dtype="float64",
    interaction_range=0.0,
    atomic_types=model.get_atomic_types(),
)

mad_explorer = mta.AtomisticModel(model.eval(), metadata, capabilities)
mad_explorer.save("mtt_mad_explorer.pt")
