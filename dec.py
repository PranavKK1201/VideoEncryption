import cv2
import numpy as np
import pickle
import os
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader

def selective_xor_encrypt_decrypt(image, key):
    key_repeated = np.frombuffer((key * (image.size // len(key) + 1)), dtype=np.uint8)[:image.size]
    key_reshaped = key_repeated.reshape(image.shape)
    # Apply XOR to the entire image
    xor_result = np.bitwise_xor(image, key_reshaped)
    return xor_result

def unscramble_pixels(image_array, scramble_indices):
    unscrambled_image = np.empty_like(image_array)
    
    for channel in range(image_array.shape[2]):  # Iterate over RGB channels
        indices = scramble_indices[channel]
        unscrambled_channel_flattened = np.empty_like(image_array[..., channel].flatten())
        unscrambled_channel_flattened[indices] = image_array[..., channel].flatten()
        unscrambled_image[..., channel] = unscrambled_channel_flattened.reshape(image_array[..., channel].shape)
        
    return unscrambled_image

# Decryption parameters
video_path = 'encrypted_pdf_video.mp4'
indices_path = 'scramble_indices.pkl'
output_folder = 'decrypted_pages'
key = b'simplekey'

os.makedirs(output_folder, exist_ok=True)

# Load metadata and video file
with open(indices_path, 'rb') as f:
    metadata = pickle.load(f)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_shape = metadata['frame_shape']
page_indices_list = metadata['page_indices']
dpi = metadata.get('dpi', 600)  # Default to 600 if not specified

# Process each frame to decrypt pages
page_number = 0
pdf_pages = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # Convert BGR to RGB (OpenCV loads as BGR)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    scramble_indices_for_page = page_indices_list[page_number]
    
    # Unscramble pixels using stored indices for this page
    unscrambled_image = unscramble_pixels(frame, scramble_indices_for_page)
    
    # XOR decrypt the unscrambled image
    decrypted_image = selective_xor_encrypt_decrypt(unscrambled_image, key)
    
    # Save as uncompressed TIFF for highest quality
    pil_image = Image.fromarray(decrypted_image)
    output_path = os.path.join(output_folder, f'page_{page_number + 1}.tiff')
    pil_image.save(output_path, compression='tiff_lzw', dpi=(dpi, dpi))
    
    # Store for PDF creation
    pdf_pages.append(output_path)
    
    print(f"Decrypted page {page_number + 1}")
    page_number += 1

cap.release()

# Create high-quality PDF using PyPDF2 and reportlab
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import PyPDF2
from io import BytesIO

def create_high_quality_pdf(image_paths, output_path, dpi=600):
    """Create a high-quality PDF from image files without compression"""
    print("Processing pdf.......")
    pdf_writer = PyPDF2.PdfWriter()
    
    for img_path in image_paths:
        img = Image.open(img_path)
        width, height = img.size
        
        # Create PDF page with the same dimensions as the image
        packet = BytesIO()
        c = canvas.Canvas(packet, pagesize=(width, height))
        c.drawImage(img_path, 0, 0, width, height, preserveAspectRatio=True)
        c.save()
        
        # Move to the beginning of the StringIO buffer
        packet.seek(0)
        new_pdf = PyPDF2.PdfReader(packet)
        pdf_writer.add_page(new_pdf.pages[0])
    
    # Save the PDF with high quality settings
    with open(output_path, 'wb') as f:
        pdf_writer.write(f)

# Create high-quality PDF
output_pdf_path = os.path.join(output_folder, "reconstructed_high_quality.pdf")
create_high_quality_pdf(pdf_pages, output_pdf_path, dpi)

print(f"High-quality PDF saved to {output_pdf_path}")

# Optionally, create a PDF directly from TIFF files using img2pdf for even better quality
try:
    import img2pdf
    
    # Create PDF with img2pdf (lossless conversion)
    with open(os.path.join(output_folder, "reconstructed_lossless.pdf"), "wb") as f:
        f.write(img2pdf.convert(pdf_pages, dpi=dpi))
    
    print(f"Lossless PDF also saved to {os.path.join(output_folder, 'reconstructed_lossless.pdf')}")
except ImportError:
    print("img2pdf not installed. Only the standard high-quality PDF was created.")
