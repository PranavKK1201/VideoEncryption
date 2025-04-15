import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image
import pickle
import os

def selective_xor_encrypt_decrypt(image, key):
    key_repeated = np.frombuffer((key * (image.size // len(key) + 1)), dtype=np.uint8)[:image.size]
    key_reshaped = key_repeated.reshape(image.shape)
    # Apply XOR to the entire image
    xor_result = np.bitwise_xor(image, key_reshaped)
    return xor_result

def scramble_pixels(image_array, key):
    seed = int.from_bytes(key, 'little') % (2**32)
    np.random.seed(seed)
    scrambled_image = np.empty_like(image_array)
    scramble_indices = []
    
    for channel in range(image_array.shape[2]):  # Iterate over RGB channels
        indices = np.arange(image_array[..., channel].size)
        np.random.shuffle(indices)
        scrambled_image[..., channel] = image_array[..., channel].flatten()[indices].reshape(image_array[..., channel].shape)
        scramble_indices.append(indices)  # Store indices for each channel
        
    return scrambled_image, scramble_indices

# Encryption parameters
pdf_path = "ac.pdf"
video_path = 'encrypted_pdf_video.mp4'
indices_path = 'scramble_indices.pkl'
key = b'simplekey'
fps = 1  # 1 frame per second

# High quality settings
dpi = 400  # Higher DPI for better quality
quality = 100  # Maximum quality for image processing

# Open PDF and initialize video writer
pdf_document = fitz.open(pdf_path)

# Get the first page to determine dimensions
first_page = pdf_document[0]
# Calculate high-resolution dimensions based on DPI
zoom_factor = dpi / 72  # Standard PDF is 72 DPI
matrix = fitz.Matrix(zoom_factor, zoom_factor)
frame_width = int(first_page.rect.width * zoom_factor)
frame_height = int(first_page.rect.height * zoom_factor)
frame_size = (frame_width, frame_height)
frame_shape = (frame_height, frame_width, 3)

# Use lossless codec for better quality
fourcc = cv2.VideoWriter_fourcc(*'RGBA')  # Lossless codec
out = cv2.VideoWriter(video_path, fourcc, fps, frame_size, isColor=True)

metadata = {'frame_shape': frame_shape, 'page_indices': [], 'dpi': dpi}

# Process each page
for page_number in range(pdf_document.page_count):
    page = pdf_document[page_number]
    
    # Render page at high resolution
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
    
    # Convert to PIL Image without compression
    img = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
    
    # Convert to numpy array for processing
    img_array = np.array(img, dtype=np.uint8)
    
    # Encrypt and scramble
    encrypted_image = selective_xor_encrypt_decrypt(img_array, key)
    scrambled_image, scramble_indices = scramble_pixels(encrypted_image, key)
    
    # Save scramble indices for decryption
    metadata['page_indices'].append(scramble_indices)
    
    # Write scrambled image to video without compression
    out.write(scrambled_image.astype(np.uint8))
    
    print(f"Processed page {page_number + 1}/{pdf_document.page_count}")

# Finalize encryption process
out.release()

# Save metadata with additional quality information
with open(indices_path, 'wb') as f:
    pickle.dump(metadata, f)

pdf_document.close()
print(f"Encryption complete. Video saved to {video_path}")
