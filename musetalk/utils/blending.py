from PIL import Image
import numpy as np
import cv2
import copy
from face_parsing import FaceParsing

fp = FaceParsing()

def get_crop_box(box, expand):
    x, y, x1, y1 = box
    x_c, y_c = (x+x1)//2, (y+y1)//2
    w, h = x1-x, y1-y
    s = int(max(w, h)//2*expand)
    crop_box = [x_c-s, y_c-s, x_c+s, y_c+s]
    return crop_box, s

def face_seg(image):
    seg_image = fp(image)
    if seg_image is None:
        print("error, no person_segment")
        return None

    seg_image = seg_image.resize(image.size)
    return seg_image

def get_image(image,face,face_box,upper_boundary_ratio = 0.5,expand=1.2):
    #print(image.shape)
    #print(face.shape)
    
    body = Image.fromarray(image[:,:,::-1])
    face = Image.fromarray(face[:,:,::-1])

    x, y, x1, y1 = face_box 
    #print(x1-x,y1-y)
    crop_box, s = get_crop_box(face_box, expand)
    x_s, y_s, x_e, y_e = crop_box
    face_position = (x, y)

    face_large = body.crop(crop_box)
    ori_shape = face_large.size

    mask_image = face_seg(face_large)
    mask_small = mask_image.crop((x-x_s, y-y_s, x1-x_s, y1-y_s))
    mask_image = Image.new('L', ori_shape, 0)
    mask_image.paste(mask_small, (x-x_s, y-y_s, x1-x_s, y1-y_s))

    # keep upper_boundary_ratio of talking area
    width, height = mask_image.size
    top_boundary = int(height * upper_boundary_ratio)
    modified_mask_image = Image.new('L', ori_shape, 0)
    modified_mask_image.paste(mask_image.crop((0, top_boundary, width, height)), (0, top_boundary))

    blur_kernel_size = int(0.1 * ori_shape[0] // 2 * 2) + 1
    mask_array = cv2.GaussianBlur(np.array(modified_mask_image), (blur_kernel_size, blur_kernel_size), 0)
    mask_image = Image.fromarray(mask_array)
    
    face_large.paste(face, (x-x_s, y-y_s, x1-x_s, y1-y_s))
    body.paste(face_large, crop_box[:2], mask_image)
    body = np.array(body)
    return body[:,:,::-1]

def get_image_prepare_material(image,face_box,upper_boundary_ratio = 0.5,expand=1.2):
    body = Image.fromarray(image[:,:,::-1])

    x, y, x1, y1 = face_box
    #print(x1-x,y1-y)
    crop_box, s = get_crop_box(face_box, expand)
    x_s, y_s, x_e, y_e = crop_box

    face_large = body.crop(crop_box)
    ori_shape = face_large.size

    mask_image = face_seg(face_large)
    mask_small = mask_image.crop((x-x_s, y-y_s, x1-x_s, y1-y_s))
    mask_image = Image.new('L', ori_shape, 0)
    mask_image.paste(mask_small, (x-x_s, y-y_s, x1-x_s, y1-y_s))

    # keep upper_boundary_ratio of talking area
    width, height = mask_image.size
    top_boundary = int(height * upper_boundary_ratio)
    modified_mask_image = Image.new('L', ori_shape, 0)
    modified_mask_image.paste(mask_image.crop((0, top_boundary, width, height)), (0, top_boundary))

    blur_kernel_size = int(0.1 * ori_shape[0] // 2 * 2) + 1
    mask_array = cv2.GaussianBlur(np.array(modified_mask_image), (blur_kernel_size, blur_kernel_size), 0)
    return mask_array,crop_box

def get_image_blending(image, face, face_box, mask_array, crop_box):
    body = image
    x, y, x1, y1 = face_box
    x_s, y_s, x_e, y_e = crop_box
    
    # Clamp crop_box to body dimensions
    body_height, body_width = body.shape[0], body.shape[1]
    x_s = max(0, x_s)
    y_s = max(0, y_s)
    x_e = min(body_width, x_e)
    y_e = min(body_height, y_e)
    
    if x_e <= x_s or y_e <= y_s:
        print(f"Invalid crop after clamping: {x_s}:{x_e}, {y_s}:{y_e} vs {body.shape}")
        return body
    
    face_large = copy.deepcopy(body[y_s:y_e, x_s:x_e])
    
    print(f"body shape: {body.shape}")
    print(f"face shape: {face.shape}")
    print(f"face_large shape: {face_large.shape}")
    print(f"face_box: {face_box}")
    print(f"crop_box (clamped): [{x_s}, {y_s}, {x_e}, {y_e}]")
    print(f"Expected face size: height={y1-y}, width={x1-x}")
    print(f"Slice indices: y={y-y_s}:{y1-y_s}, x={x-x_s}:{x1-x_s}")
    
    expected_height = y1 - y
    expected_width = x1 - x
    if expected_width <= 0 or expected_height <= 0:
        raise ValueError(f"Invalid face_box: width={expected_width}, height={expected_height}")
    slice_height = y1 - y_s - (y - y_s)
    slice_width = x1 - x_s - (x - x_s)
    if slice_width <= 0 or slice_height <= 0:
        print(f"Invalid slice after clamping: height={slice_height}, width={slice_width}")
        return body
    if x-x_s < 0 or x1-x_s > face_large.shape[1] or y-y_s < 0 or y1-y_s > face_large.shape[0]:
        print(f"Slice out of bounds after clamping: {x-x_s}:{x1-x_s}, {y-y_s}:{y1-y_s} vs {face_large.shape}")
        return body

    if face.shape[0] != expected_height or face.shape[1] != expected_width:
        face = cv2.resize(face, (expected_width, expected_height), interpolation=cv2.INTER_AREA)
    
    if len(face.shape) == 2:
        face = cv2.cvtColor(face, cv2.COLOR_GRAY2BGR)
    elif face.shape[2] != face_large.shape[2]:
        face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR if face_large.shape[2] == 3 else cv2.COLOR_BGR2RGB)

    face_large[y-y_s:y1-y_s, x-x_s:x1-x_s] = face

    # Handle mask_array
    if len(mask_array.shape) == 2 or mask_array.shape[-1] == 1:
        mask_image = mask_array
    else:
        mask_image = cv2.cvtColor(mask_array, cv2.COLOR_BGR2GRAY)
    
    # Resize mask_image to match face_large
    if mask_image.shape != (face_large.shape[0], face_large.shape[1]):
        mask_image = cv2.resize(mask_image, (face_large.shape[1], face_large.shape[0]), interpolation=cv2.INTER_AREA)
    
    mask_image = (mask_image / 255).astype(np.float32)
    weights2 = 1 - mask_image
    
    # Verify sizes before blending
    if (face_large.shape[:2] != body[y_s:y_e, x_s:x_e].shape[:2] or
        face_large.shape[:2] != mask_image.shape or
        face_large.shape[:2] != weights2.shape):
        print(f"Size mismatch: face_large={face_large.shape}, body_slice={body[y_s:y_e, x_s:x_e].shape}, "
              f"mask_image={mask_image.shape}, weights2={weights2.shape}")
        return body

    body[y_s:y_e, x_s:x_e] = cv2.blendLinear(face_large, body[y_s:y_e, x_s:x_e], mask_image, weights2)

    return body