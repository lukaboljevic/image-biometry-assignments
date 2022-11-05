def IoU(box1, box2):
    """
    Calculate the Intersection over Union between two bounding boxes.
    
    A box is given with a list 
    [topleft_x, topleft_y, bottomright_x, bottomright_y]
    """
	# Determine the (x, y)-coordinates of the intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

	# Compute the area of intersection rectangle
    intersection = (x_right - x_left + 1) * (y_bottom - y_top + 1)

    # Compute the area of both boxes
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # Compute the intersection over union
    iou = intersection / float(box1_area + box2_area - intersection)

    assert iou >= 0.0
    assert iou <= 1.0
    return iou