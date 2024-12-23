# Text message
message = "HiddenWatermark2023"

# Convert message to bytes
watermark_bits = message.encode('utf-8')

# Define the file path
watermark_path = "text_watermark.bin"

# Write to file
with open(watermark_path, "wb") as f:
    f.write(watermark_bits)

print(f"Text-based watermark file created at {watermark_path}")