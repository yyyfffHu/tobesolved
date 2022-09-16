

result_list = []

for result in result_list:
    gt_box = result["text_polys"]
    gt_tag = result["text_tags"]
    det_boxes = result["pred_polys"]
    det_tags = result["pred_tags"]

    #gt_box的结构为（？，4，2）
    #gt_tag的结构为（0，1，2，...,）
    #det_boxes理论上与gt一样
