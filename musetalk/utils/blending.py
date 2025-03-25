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
    
    face_large = copy.deepcopy(body[y_s:y_e, x_s:x_e])
    
    # Log everything
    print(f"face shape: {face.shape}")
    print(f"face_large shape: {face_large.shape}")
    print(f"face_box: {face_box}")
    print(f"crop_box: {crop_box}")
    print(f"Expected face size: height={y1-y}, width={x1-x}")
    print(f"Slice indices: y={y-y_s}:{y1-y_s}, x={x-x_s}:{x1-x_s}")
    
    # Validate coordinates
    expected_height = y1 - y
    expected_width = x1 - x
    if expected_width <= 0 or expected_height <= 0:
        raise ValueError(f"Invalid face_box: width={expected_width}, height={expected_height}")
    slice_height = y1 - y_s - (y - y_s)
    slice_width = x1 - x_s - (x - x_s)
    if slice_width <= 0 or slice_height <= 0:
        raise ValueError(f"Invalid slice: height={slice_height}, width={slice_width}")
    if x-x_s < 0 or x1-x_s > face_large.shape[1] or y-y_s < 0 or y1-y_s > face_large.shape[0]:
        raise ValueError(f"Slice out of bounds: {x-x_s}:{x1-x_s}, {y-y_s}:{y1-y_s} vs {face_large.shape}")

    # Resize face if necessary
    if face.shape[0] != expected_height or face.shape[1] != expected_width:
        face = cv2.resize(face, (expected_width, expected_height), interpolation=cv2.INTER_AREA)
    
    # Channel consistency
    if len(face.shape) == 2:
        face = cv2.cvtColor(face, cv2.COLOR_GRAY2BGR)
    elif face.shape[2] != face_large.shape[2]:
        face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR if face_large.shape[2] == 3 else cv2.COLOR_BGR2RGB)

    face_large[y-y_s:y1-y_s, x-x_s:x1-x_s] = face

    if len(mask_array.shape) == 2 or mask_array.shape[-1] == 1:
        mask_image = mask_array
    else:
        mask_image = cv2.cvtColor(mask_array, cv2.COLOR_BGR2GRAY)

    mask_image = (mask_image / 255).astype(np.float32)
    body[y_s:y_e, x_s:x_e] = cv2.blendLinear(face_large, body[y_s:y_e, x_s:x_e], mask_image, 1 - mask_image)

    return body