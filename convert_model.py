from tensorflow.keras.models import load_model

# Purana model load karo
model = load_model("stock_model.h5", compile=False)

# Naya format mein save karo
model.save("stock_model.keras")
print("✅ Model successfully converted to new format.")
