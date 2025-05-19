"""
Sample data utility for ONNX model inference.
This module provides functions to download and prepare sample images for inference.
"""

import os
import cv2
import numpy as np
import urllib.request
from pathlib import Path

# ImageNet class labels (top 100 common classes)
IMAGENET_CLASSES = {
    0: 'tench, Tinca tinca',
    1: 'goldfish, Carassius auratus',
    2: 'great white shark, white shark',
    3: 'tiger shark, Galeocerdo cuvieri',
    4: 'hammerhead, hammerhead shark',
    5: 'electric ray, crampfish',
    6: 'stingray',
    7: 'cock',
    8: 'hen',
    9: 'ostrich, Struthio camelus',
    10: 'brambling, Fringilla montifringilla',
    11: 'goldfinch, Carduelis carduelis',
    12: 'house finch, linnet, Carpodacus mexicanus',
    13: 'junco, snowbird',
    14: 'indigo bunting, indigo finch',
    15: 'robin, American robin, Turdus migratorius',
    16: 'bulbul',
    17: 'jay',
    18: 'magpie',
    19: 'chickadee',
    20: 'water ouzel, dipper',
    21: 'kite',
    22: 'bald eagle, American eagle',
    23: 'vulture',
    24: 'great grey owl, great gray owl',
    25: 'European fire salamander',
    26: 'common newt, Triturus vulgaris',
    27: 'eft',
    28: 'spotted salamander',
    29: 'axolotl, mud puppy',
    30: 'bullfrog, Rana catesbeiana',
    31: 'tree frog, tree-frog',
    32: 'tailed frog',
    33: 'loggerhead, loggerhead turtle',
    34: 'leatherback turtle',
    35: 'mud turtle',
    36: 'terrapin',
    37: 'box turtle, box tortoise',
    38: 'banded gecko',
    39: 'common iguana, iguana',
    40: 'American chameleon, anole',
    41: 'whiptail, whiptail lizard',
    42: 'agama',
    43: 'frilled lizard',
    44: 'alligator lizard',
    45: 'Gila monster, Heloderma suspectum',
    46: 'green lizard, Lacerta viridis',
    47: 'African chameleon, Chamaeleo chamaeleon',
    48: 'Komodo dragon',
    49: 'African crocodile, Nile crocodile',
    50: 'American alligator, Alligator mississipiensis',
    51: 'triceratops',
    52: 'thunder snake, worm snake',
    53: 'ringneck snake, ring-necked snake',
    54: 'hognose snake, puff adder, sand viper',
    55: 'green snake, grass snake',
    56: 'king snake, kingsnake',
    57: 'garter snake, grass snake',
    58: 'water snake',
    59: 'vine snake',
    60: 'night snake, Hypsiglena torquata',
    61: 'boa constrictor, Constrictor constrictor',
    62: 'rock python, rock snake, Python sebae',
    63: 'Indian cobra, Naja naja',
    64: 'green mamba',
    65: 'sea snake',
    66: 'horned viper, cerastes',
    67: 'diamondback, diamondback rattlesnake',
    68: 'sidewinder, horned rattlesnake',
    69: 'trilobite',
    70: 'harvestman, daddy longlegs',
    71: 'scorpion',
    72: 'black and gold garden spider',
    73: 'barn spider, Araneus cavaticus',
    74: 'garden spider, Aranea diademata',
    75: 'black widow, Latrodectus mactans',
    76: 'tarantula',
    77: 'wolf spider, hunting spider',
    78: 'tick',
    79: 'centipede',
    80: 'black grouse',
    81: 'ptarmigan',
    82: 'ruffed grouse, partridge',
    83: 'prairie chicken, prairie grouse',
    84: 'peacock',
    85: 'quail',
    86: 'partridge',
    87: 'African grey, African gray',
    88: 'macaw',
    89: 'sulphur-crested cockatoo',
    90: 'lorikeet',
    91: 'coucal',
    92: 'bee eater',
    93: 'hornbill',
    94: 'hummingbird',
    95: 'jacamar',
    96: 'toucan',
    97: 'drake',
    98: 'red-breasted merganser',
    99: 'goose',
}

# Common ImageNet classes that might be predicted for our sample images
# Adding more classes beyond the basic 100 in IMAGENET_CLASSES
EXTENDED_CLASSES = {
    281: 'tabby cat',
    282: 'tiger cat',
    283: 'Persian cat',
    284: 'Siamese cat, Siamese',
    285: 'Egyptian cat',
    287: 'lynx, catamount',

    151: 'Chihuahua',
    152: 'Japanese Spaniel',
    153: 'Maltese dog, Maltese terrier',
    154: 'Pekinese, Pekingese',
    155: 'Shih-Tzu',
    156: 'Blenheim spaniel',
    157: 'papillon',
    158: 'toy terrier',
    159: 'Rhodesian ridgeback',
    160: 'Afghan hound, Afghan',
    161: 'basset, basset hound',
    162: 'beagle',
    163: 'bloodhound, sleuthhound',

    407: 'ambulance',
    436: 'beach wagon, station wagon',
    468: 'cab, hack, taxi, taxicab',
    511: 'convertible',
    627: 'limousine, limo',
    656: 'minivan',
    661: 'Model T',
    751: 'racer, race car, racing car',
    817: 'sports car, sport car',

    985: 'daisy',
    986: 'yellow lady\'s slipper',
    987: 'corn',
    988: 'acorn',
    989: 'rose hip',
    990: 'horse chestnut seed',
    991: 'coral fungus',
    992: 'agaric',
    993: 'gyromitra',
    994: 'stinkhorn mushroom',
    995: 'earth star',
    996: 'hen-of-the-woods',
    997: 'bolete',
    998: 'ear, spike, capitulum',
    999: 'toilet tissue, toilet paper, bathroom tissue'
}

# Sample image URLs (public domain or CC licensed images)
SAMPLE_IMAGES = {
    'cat': 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/481px-Cat03.jpg',
    'dog': 'https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Labrador_on_Quantock_%282175262184%29.jpg/640px-Labrador_on_Quantock_%282175262184%29.jpg',
    'bird': 'https://upload.wikimedia.org/wikipedia/commons/thumb/4/45/Eopsaltria_australis_-_Mogo_Campground.jpg/640px-Eopsaltria_australis_-_Mogo_Campground.jpg',
    'car': 'https://upload.wikimedia.org/wikipedia/commons/thumb/5/5a/1970_AMC_Rebel_SST_coupe_front.jpg/640px-1970_AMC_Rebel_SST_coupe_front.jpg',
    'flower': 'https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Sunflower_from_Silesia2.jpg/640px-Sunflower_from_Silesia2.jpg',
}

def download_sample_images(output_dir='ML/sample_images'):
    """Download sample images for inference testing"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    downloaded_paths = {}

    for name, url in SAMPLE_IMAGES.items():
        try:
            # Define output path
            output_path = os.path.join(output_dir, f"{name}.jpg")

            # Download the image if it doesn't exist
            if not os.path.exists(output_path):
                print(f"Downloading {name} image...")
                urllib.request.urlretrieve(url, output_path)
                print(f"Downloaded to {output_path}")
            else:
                print(f"Image {name} already exists at {output_path}")

            downloaded_paths[name] = output_path
        except Exception as e:
            print(f"Error downloading {name} image: {e}")

    return downloaded_paths

def preprocess_image(image_path, input_shape=(224, 224), normalize=True, to_rgb=True):
    """
    Preprocess an image for neural network inference

    Args:
        image_path: Path to the image file
        input_shape: Target shape (height, width)
        normalize: Whether to normalize pixel values to [0,1]
        to_rgb: Whether to convert BGR to RGB (OpenCV loads as BGR)

    Returns:
        Preprocessed image as numpy array in NCHW format
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Resize to target shape
    img = cv2.resize(img, (input_shape[1], input_shape[0]))

    # Convert BGR to RGB if needed
    if to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize pixel values
    if normalize:
        img = img.astype(np.float32) / 255.0

    # Transpose to NCHW format (batch, channels, height, width)
    img = img.transpose(2, 0, 1)

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    return img

def get_class_name(class_id):
    """Get class name from class ID"""
    # First check our extended classes for common predictions
    if class_id in EXTENDED_CLASSES:
        return EXTENDED_CLASSES[class_id]
    # Then check the original 100 classes
    elif class_id in IMAGENET_CLASSES:
        return IMAGENET_CLASSES[class_id]
    else:
        return f"Unknown class {class_id}"

def get_top_predictions(output, top_k=5):
    """
    Get top-k predictions from model output

    Args:
        output: Model output (numpy array)
        top_k: Number of top predictions to return

    Returns:
        List of (class_id, class_name, probability) tuples
    """
    # Ensure output is a numpy array
    if isinstance(output, list):
        output = output[0]

    # Get top-k indices
    top_indices = np.argsort(output.flatten())[-top_k:][::-1]

    # Get probabilities for top indices
    top_probs = output.flatten()[top_indices]

    # Create result list
    results = []
    for i, idx in enumerate(top_indices):
        results.append((int(idx), get_class_name(int(idx)), float(top_probs[i])))

    return results

if __name__ == "__main__":
    # Test the module
    images = download_sample_images()
    print(f"Downloaded {len(images)} sample images")

    # Test preprocessing
    for name, path in images.items():
        img = preprocess_image(path)
        print(f"Preprocessed {name}: shape={img.shape}, min={img.min()}, max={img.max()}")
