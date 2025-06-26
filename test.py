
def compute_iou(bbox_1: torch.Tensor, bbox_2: torch.Tensor) -> torch.Tensor:
    assert bbox_1.shape == bbox_2.shape

    ########################################################################
    # TODO:                                                                #
    # Compute the intersection over union (IoU) for two batches of         #
    # bounding boxes, each of shape (B, 4). The result should be a tensor  #
    # of shape (B,).                                                       #
    # NOTE: the format of the bounding boxes is (ltrb), meaning            #
    # (left edge, top edge, right edge, bottom edge). Remember the         #
    # orientation of the image coordinates.                                #
    # NOTE: First calculate the intersection and use this to compute the   #
    # union                                                                #
    # iou = ...                                                            #
    ########################################################################
    left = torch.max(bbox_1[:, 0], bbox_2[:, 0])
    top = torch.max(bbox_1[:, 1], bbox_2[:, 1])
    right = torch.min(bbox_1[:, 2], bbox_2[:, 2])
    bottom = torch.min(bbox_1[:, 3], bbox_2[:, 3])

    intersection_area = torch.clamp(right - left, min=0) * torch.clamp(bottom - top, min=0)

    bbox_1_area = (bbox_1[:, 2] - bbox_1[:, 0]) * (bbox_1[:, 3] - bbox_1[:, 1])
    bbox_2_area = (bbox_2[:, 2] - bbox_2[:, 0]) * (bbox_2[:, 3] - bbox_2[:, 1])

    union_area = bbox_1_area + bbox_2_area - intersection_area

    iou = intersection_area / union_area
    pass

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################

    return iou


def compute_image_gradient(images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # images B x H x W

    ########################################################################
    # TODO:                                                                #
    # Compute the 2-dimenational gradient for a given grey image of size   #
    # B x H x W. The return values of this function should be the norm and #
    # the angle of this gradient vector.                                   #
    # NOTE: first, calculate the gradient in x and y direction             #
    # (you will need add padding to the image boundaries),                 #
    # then, compute the vector norm and angle.                             #
    # The angle of a given gradient angle is defined                       #
    # in degrees (range=0.,,,.360).                                        #
    # NOTE: The angle is defined counter-clockwise angle between the       #
    # gradient and the unit vector along the x-axis received from atan2.   #
    ########################################################################
    # images B x H x W

    padded_images = nn.functional.pad(images, (1, 1, 1, 1), mode="replicate")  # Padding by 1 on each side

    gradient_x = (padded_images[:, :, 2:] - padded_images[:, :, :-2])
    gradient_y = (padded_images[:, 2:, :] - padded_images[:, :-2, :])

    gradient_x = gradient_x[:, 1:-1, :]
    gradient_y = gradient_y[:, :, 1:-1]

    norm = torch.sqrt(gradient_x ** 2 + gradient_y ** 2)

    angle = torch.atan2(gradient_y, gradient_x) * (180 / torch.pi)
    angle[angle < 0] += 360

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################

    return norm, angle

def sliding_window_detection(
    image: torch.Tensor, model: nn.Module, patch_size: Tuple[int, int], stride: int = 1
) -> torch.Tensor:
    grayscale_image = transforms.functional.rgb_to_grayscale(image).float()
    block_size = 8
    num_bins = 9
    H_p, W_p = patch_size[0], patch_size[1]
    ########################################################################
    # TODO:                                                                #
    # Perform person detection in a sliding window manner with HoG         #
    # features and a trained classification model. For this                #
    #                                                                      #
    # - get patches of the grayscale image by using the patch_size and     #
    # stride parameters. Be aware of the stride parameter and don't pad    #
    # the image.                                                           #
    #                                                                      #
    # - compute the HoG features for each patch. The function call is      #
    # hog_features = HoG(patches, block_size, num_bins), where             #
    # block_size and num_bins are hard coded to 8 and 9, respectively.     #
    # Use the HoG function on the batch of grey valued patches of shape    #
    # (B, patch_size[0], patch_size[1]) to compute the flattened HoG       #
    # features.                                                            #
    #                                                                      #
    # - use the model for prediction.                                      #
    #                                                                      #
    # - reshape the output of the model to have 2-spatial dimensions       #
    # and classification scores as values.                                 #
    #                                                                      #
    # Input:                                                               #
    # A single input image of size (3, H, W).                              #
    # The stride and patch_size for of the sliding window operation.       #
    # (With the provided parameters, the hog_features fit                  #
    # the expected input size of the classifier)                           #
    # The classification model, which takes a batch of flattened feature   #
    # patches (hog_features is already flattened)                          #
    # Output:                                                              #
    # An collection if classified patches. The ouput shape should be       #
    # (floor((H-(patch_size[0]-1)-1)/stride+1),                            #
    # floor((W-(patch_size[1]-1)-1)/stride+1)).                            #
    #                                                                      #
    ########################################################################
    output_height = floor((image.shape[1] - (H_p - 1) - 1) / stride + 1)
    output_width = floor((image.shape[2] - (W_p - 1) - 1) / stride + 1)

    patch_classifications = []


    for y in range(0, image.shape[1] - H_p + 1, stride):
        for x in range(0, image.shape[2] - W_p + 1, stride):
            # Extract the patch
            patch = grayscale_image[:, y:y+H_p, x:x+W_p]

            hog_features = HoG(patch, block_size, num_bins)

            prediction = model(hog_features)

            patch_classifications.append(prediction)

    detection_image = torch.stack(patch_classifications).reshape(output_height, output_width)
    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################

    return detection_image


def non_maximum_suppression(bboxes: torch.Tensor, scores: torch.Tensor, threshold: float) -> torch.Tensor:
    ########################################################################
    # TODO:                                                                #
    # Compute the non maximum suppression                                  #
    # Input:                                                               #
    # bounding boxes of shape B,4                                          #
    # scores of shape B                                                    #
    # threshold for iou: if the overlap is bigger, only keep one of the    #
    # bboxes                                                               #
    # Output:                                                              #
    # bounding boxes of shape B_,4                                         #
    ########################################################################

    sorted_indices = torch.argsort(scores, descending=True)
    bboxes = bboxes[sorted_indices]

    selected_bboxes = []

    while len(bboxes) > 0:
        # Select the bounding box with the highest score
        top_bbox = bboxes[0]
        selected_bboxes.append(top_bbox)

        if len(bboxes) == 1:
            break

        # Compute IoU of the top box with the rest
        ious = compute_iou(top_bbox.unsqueeze(0), bboxes[1:])

        # Keep only those boxes whose IoU is below the threshold
        keep_indices = (ious < threshold).nonzero().squeeze(1)
        bboxes = bboxes[1:][keep_indices]

    # Convert the list of selected bounding boxes to a tensor
    bboxes_nms = torch.stack(selected_bboxes)

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################

    return bboxes_nms


import numpy as np
from collections import defaultdict

def iou_3d_aabb(box1, box2):
    """Axis-aligned 3D IoU."""
    def get_bounds(box):
        center = np.array(box['center'])
        size = np.array(box['size']) / 2
        min_corner = center - size
        max_corner = center + size
        return min_corner, max_corner

    min1, max1 = get_bounds(box1)
    min2, max2 = get_bounds(box2)

    inter_min = np.maximum(min1, min2)
    inter_max = np.minimum(max1, max2)
    inter_dim = np.maximum(inter_max - inter_min, 0)
    inter_vol = np.prod(inter_dim)

    vol1 = np.prod(max1 - min1)
    vol2 = np.prod(max2 - min2)

    union = vol1 + vol2 - inter_vol + 1e-6
    return inter_vol / union

def evaluate_mot(data, iou_thresh=0.5):
    prev_gt_to_pred = dict()
    TP, FP, FN, IDSW, total_iou = 0, 0, 0, 0, 0
    total_gt = 0

    for frame_id in sorted(data.keys()):
        gt_frame = data[frame_id]['gt']
        pred_frame = data[frame_id]['pred']
        matches = []
        unmatched_gt = set(range(len(gt_frame)))
        unmatched_pred = set(range(len(pred_frame)))

        # Compute pairwise IoUs
        iou_matrix = np.zeros((len(gt_frame), len(pred_frame)))
        for i, gt in enumerate(gt_frame):
            for j, pred in enumerate(pred_frame):
                iou_matrix[i, j] = iou_3d_aabb(gt, pred)

        # Greedy matching
        while True:
            i, j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            if iou_matrix[i, j] < iou_thresh:
                break
            gt_id = gt_frame[i]['id']
            pred_id = pred_frame[j]['id']
            matches.append((gt_id, pred_id, iou_matrix[i, j]))
            unmatched_gt.discard(i)
            unmatched_pred.discard(j)
            iou_matrix[i, :] = -1
            iou_matrix[:, j] = -1

        total_gt += len(gt_frame)
        TP += len(matches)
        FP += len(unmatched_pred)
        FN += len(unmatched_gt)
        total_iou += sum([m[2] for m in matches])

        # Count ID switches
        for gt_id, pred_id, _ in matches:
            if gt_id in prev_gt_to_pred:
                if prev_gt_to_pred[gt_id] != pred_id:
                    IDSW += 1
            prev_gt_to_pred[gt_id] = pred_id

    mota = 1 - (FN + FP + IDSW) / (total_gt + 1e-6)
    motp = total_iou / (TP + 1e-6)
    return {'MOTA': mota, 'MOTP': motp, 'FP': FP, 'FN': FN, 'IDSW': IDSW, 'TP': TP}

# ✅ Example test
example_data = {
    0: {
        'gt': [{'id': 1, 'center': [0,0,0], 'size': [2,2,2], 'yaw': 0}],
        'pred': [{'id': 10, 'center': [0.1,0.1,0], 'size': [2,2,2], 'yaw': 0}]
    },
    1: {
        'gt': [{'id': 1, 'center': [0.2,0,0], 'size': [2,2,2], 'yaw': 0}],
        'pred': [{'id': 11, 'center': [0.3,0.1,0], 'size': [2,2,2], 'yaw': 0}]
    }
}

results = evaluate_mot(example_data)
print("Tracking Evaluation:", results)
