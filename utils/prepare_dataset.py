import json
import random
import shutil
from shutil import copyfile, move
import os

def combine_annotation_files(train_json_path, val_json_path, combined_json_path):
    with open(train_json_path, 'r') as train_file, open(val_json_path, 'r') as val_file:
        train_data = json.load(train_file)
        val_data = json.load(val_file)

        combined_data = train_data

        # Combine images
        combined_data['images'] = train_data['images'] + val_data['images']

        # Combine annotations
        combined_data['annotations'] = train_data['annotations'] + val_data['annotations']

    with open(combined_json_path, 'w') as combined_file:
        json.dump(combined_data, combined_file)

def combine_scene_graph_files(train_json_path, val_json_path, combined_json_path):
    with open(train_json_path, 'r') as train_file, open(val_json_path, 'r') as val_file:
        train_data = json.load(train_file)
        val_data = json.load(val_file)

        merge_json = {**train_data, **val_data}
        # merge_json = json.dumps(merge_json)

    with open(combined_json_path, 'w') as combined_file:
        json.dump(merge_json, combined_file)

def combined_train_validation_sets(train_path, val_path, combined_images_path, combined_scene_graph_path):

    # Read the content of the sg.json file
    with open(combined_scene_graph_path, 'r') as sg_file:
        sg_data = json.load(sg_file)

        # Iterate over the image names in the sg.json file
        train_images = os.listdir(train_path)
        val_images = os.listdir(val_path)
        sg = list(sg_data.keys())
        print(len(train_images))
        print(len(val_images))
        print(len(sg))

        for train_img in train_images:
            if train_img in sg:
                shutil.copy(os.path.join(train_path, train_img), combined_images_path)
                sg.remove(train_img)
        for val_img in val_images:
            if val_img in sg:
                shutil.copy(os.path.join(val_path, val_img), combined_images_path)
                sg.remove(val_img)
        print(len(sg))

def split_dataset1(combined_json_path, combined_images_path, val_output_path, test_output_path, num_val, num_test):
    with open(combined_json_path, 'r') as combined_file:
        combined_data = json.load(combined_file)

    images = combined_data['images']
    random.shuffle(images)

    val_images = images[:num_val]
    test_images = images[num_val:num_val + num_test]

    val_image_ids = set(img['id'] for img in val_images)
    test_image_ids = set(img['id'] for img in test_images)

    val_annotations = [ann for ann in combined_data['annotations'] if ann['image_id'] in val_image_ids]
    test_annotations = [ann for ann in combined_data['annotations'] if ann['image_id'] in test_image_ids]

    val_data = {'images': val_images, 'annotations': val_annotations}
    test_data = {'images': test_images, 'annotations': test_annotations}

    with open(os.path.join(val_output_path, 'captions_val5000.json'), 'w') as val_file:
        json.dump(val_data, val_file)

    with open(os.path.join(test_output_path, 'captions_test5000.json'), 'w') as test_file:
        json.dump(test_data, test_file)

    for img in val_images:
        img_name = img['file_name']
        move(os.path.join(combined_images_path, img_name), os.path.join(val_output_path, 'images', img_name))

    for img in test_images:
        img_name = img['file_name']
        move(os.path.join(combined_images_path, img_name), os.path.join(test_output_path, 'images', img_name))

def split_dataset(image_folder, sg_json_path, caption_json_path, test_output_path, val_output_path, num_test, num_val):
    # Read the content of the sg.json file
    with open(sg_json_path, 'r') as sg_file:
        sg_data = json.load(sg_file)

    # Get the list of images with information in sg.json
    available_images = list(sg_data.keys())

    # Shuffle the available images
    random.shuffle(available_images)

    # Select images for testing and validation
    test_images = available_images[:num_test]
    val_images = available_images[num_test:num_test+num_val]

    # Move the selected test images to the test output path
    for image_name in test_images:
        image_path = os.path.join(image_folder, image_name)
        output_path = os.path.join(os.path.join(test_output_path, 'images'), image_name)
        shutil.move(image_path, output_path)

    # Move the selected validation images to the validation output path
    for image_name in val_images:
        image_path = os.path.join(image_folder, image_name)
        output_path = os.path.join(os.path.join(val_output_path, 'images'), image_name)
        shutil.move(image_path, output_path)

    # Create test.json with information for the selected test images
    test_data = {image_name: sg_data[image_name] for image_name in test_images}
    with open(os.path.join(test_output_path, 'test5000_Detected_Scene_Graphs.json'), 'w') as test_file:
        json.dump(test_data, test_file)

    # Create val.json with information for the selected validation images
    val_data = {image_name: sg_data[image_name] for image_name in val_images}
    with open(os.path.join(val_output_path, 'val5000_Detected_Scene_Graphs.json'), 'w') as val_file:
        json.dump(val_data, val_file)

    # # Create val5000_caption2014.json with captions for the selected validation images
    # with open(caption_json_path, 'r') as caption_file:
    #     caption_data = json.load(caption_file)
    # val5000_caption_data = [caption for caption in caption_data if caption['image_id'] in val_images]
    # with open(os.path.join(val_output_path, 'captions_val50002014.json'), 'w') as val_caption_file:
    #     json.dump(val5000_caption_data, val_caption_file)

    # # Create test5000_caption2014.json with captions for the selected test images
    # test5000_caption_data = [caption for caption in caption_data if caption['image_id'] in test_images]
    # with open(os.path.join(test_output_path, 'caption_test50002014.json'), 'w') as test_caption_file:
    #     json.dump(test5000_caption_data, test_caption_file)

def make_val_test_annotations(image_folder_path, original_json_path, filtered_json_path):
    with open(original_json_path, 'r') as json_file:
        original_data = json.load(json_file)

    # Get the list of image names in your folder
    image_names = os.listdir(image_folder_path)

    # Filter the annotations and images
    filtered_annotations = []
    filtered_images = []
    image_ids = set()

    for annotation in original_data['annotations']:
        image_id = annotation['image_id']
        image_name = 'COCO_train2014_{:012d}.jpg'.format(image_id)
        image_name2 = 'COCO_val2014_{:012d}.jpg'.format(image_id)
        if image_name in image_names or image_name2 in image_names:
            filtered_annotations.append(annotation)
            image_ids.add(image_id)

    for image in original_data['images']:
        if image['id'] in image_ids:
            filtered_images.append(image)

    # Create the filtered data object
    filtered_data = {
        'images': filtered_images,
        'annotations': filtered_annotations,
        'info': original_data['info'],
        'licenses': original_data['licenses']
    }

    # Save the filtered data to a new JSON file
    with open(filtered_json_path, 'w') as json_file:
        json.dump(filtered_data, json_file)


if __name__ == '__main__':


    # Split dataset, take 5000 val image and 5000 test image
    print('split dataset for validation and test...')
    image_val_folder = 'datasets/images/val_resized_2014'
    val_sg_path = 'datasets/SG/Val_Detected_Scene_Graphs.json'
    val_ann_path = 'datasets/annotations/captions_val2014.json'
    test_output_path = 'datasets/images/test5000'
    val_output_path = 'datasets/images/val5000'
    test_size = 5000
    val_size =5000

    if not os.path.exists(val_output_path):
        os.makedirs(val_output_path)
        if not os.path.exists(os.path.join(val_output_path, 'images')):
            os.makedirs(os.path.join(val_output_path, 'images'))

    if not os.path.exists(test_output_path):
        os.makedirs(test_output_path)
        if not os.path.exists(os.path.join(test_output_path, 'images')):
            os.makedirs(os.path.join(test_output_path, 'images'))

    split_dataset(image_val_folder, val_sg_path, val_ann_path, test_output_path, val_output_path, test_size, val_size)

    # make captions_{val_test}50002014.json annotation
    val_image_folder_path = 'datasets/images/val5000/images'
    test_image_folder_path = 'datasets/images/test5000/images'
    original_json_path = 'datasets/annotations/captions_val2014.json'
    val_filtered_json_path = 'datasets/images/val5000/captions_val50002014.json'
    test_filtered_json_path = 'datasets/images/test5000/captions_test50002014.json'

    print('making val5000 annotations file...')
    make_val_test_annotations(val_image_folder_path, original_json_path, val_filtered_json_path)

    print('making test5000 annotations file...')
    make_val_test_annotations(test_image_folder_path, original_json_path, test_filtered_json_path)

    # Combined annotation files
    print('Combined annotation files...')
    train_ann_json_path = 'datasets/annotations/captions_train2014.json'
    val_ann_json_path = 'datasets/annotations/captions_val2014.json'
    combined_ann_json_path = 'datasets/annotations/combined_annotations.json'
    combine_annotation_files(train_ann_json_path, val_ann_json_path, combined_ann_json_path)
    
    # Combine train and validation scene graphs dataset
    print('combined scene graph files...')
    train_sg_json_path = 'datasets/SG/Train_Detected_Scene_Graphs.json'
    val_sg_json_path = 'datasets/SG/Val_Detected_Scene_Graphs.json'
    combined_sg_json_path = 'datasets/SG/combined_Detected_Scene_Graphs.json'
    combine_scene_graph_files(train_sg_json_path, val_sg_json_path, combined_sg_json_path)

    # Combine train and validation sets
    train_image_path = 'datasets/images/train_resized_2014'
    val_image_path = 'datasets/images/val_resized_2014'
    combined_json_path = 'datasets/annotations/combined_annotations.json'
    combined_images_path = 'datasets/images/combined_images'


    if not os.path.exists(combined_images_path):
        os.makedirs(combined_images_path)
        # combined train and validation images just they have scene graphs
    print('combined train and validation images...')
    combined_train_validation_sets(train_image_path, val_image_path, combined_images_path, combined_sg_json_path)

    combined_image_folder_path = 'datasets/images/combined_images'
    original_json_path = 'datasets/annotations/combined_annotations.json'
    combined_filtered_json_path = 'datasets/annotations/filtered_annotations.json'
    print('making combined annotations file...')
    make_val_test_annotations(combined_image_folder_path, original_json_path, combined_filtered_json_path)