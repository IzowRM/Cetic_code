from PIL import ImageOps


def pad_to_224_width(img):
    # img est une image PIL
    w, h = img.size          # attention: PIL = (width, height)
    assert h == 224, f"Hauteur attendue = 224, mais trouvé {h}"
    if w == 224:
        return img

    diff = 224 - w
    pad_left = diff // 2
    pad_right = diff - pad_left

    # border = (left, top, right, bottom)
    img_padded = ImageOps.expand(img, border=(pad_left, 0, pad_right, 0), fill=0)
    return img_padded