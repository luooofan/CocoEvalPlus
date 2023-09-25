import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from COCOevalEveryClass import COCOevalEveryClass


class COCOevalPlus:
    """
    This is a high-level encapsulation for evaluating detection on the Microsoft COCO dataset. 
    It can also be seen as a plus version of pycocotools.cocoeval.COCOeval.

    Features: 
    1. confusion matrix
    2. every class metrics(ap ar)
    3. summary metrics(map mar)

    The usage for COCOevalPlus is as follows:
        srcjson = ..., resjson = ...       # specify the source annotation json filepath and the results json filepath
        E = COCOevalPlus(srcjson, resjson) # initialize COCOevalPlus object
        E.coco_eval.params.recThrs = ...   # set parameters as desired
        map = E.getMapMar()                # display summary metrics of results
        cm = E.getConfusionMatrix()        # compute the confusion matrix

    The way to compute confusion matrix of one image:
    1. Extracts the ground-truth boxes and classes, along with the detected boxes, classes, and scores.
    2. Only detections with a score greater or equal than CONFIDENCE_THRESHOLD(0.5) are considered. 
       Anything that’s under this value is discarded.
    3. For each ground-truth box, the algorithm generates the IoU (Intersection over Union) with every detected box. 
       A match is found if both boxes have an IoU greater or equal than IOU_THRESHOLD(0.5).
    4. The list of matches is pruned to remove duplicates (ground-truth boxes that match with more than one detection box or vice versa). 
       If there are duplicates, the best match (greater IoU) is always selected.
    5. The confusion matrix is updated to reflect the resulting matches between ground-truth and detections.
    6. Objects that are part of the ground-truth but weren’t detected are counted in the last column of the matrix (in the row corresponding to the ground-truth class). 
       Objects that were detected but aren’t part of the confusion matrix are counted in the last row of the matrix (in the column corresponding to the detected class).

    References:
    https://github.com/svpino/tf_object_detection_cm/blob/master/confusion_matrix.py

    """

    CONFIDENCE_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.5

    def __init__(self, srcjson, resjson) -> None:
        self.sources = COCO(srcjson)
        self.results = self.sources.loadRes(resjson)
        # self.coco_eval = COCOeval(self.sources, self.results, 'bbox')
        self.coco_eval = COCOevalEveryClass(self.sources, self.results, 'bbox')
        self.coco_eval._prepare()

    def _compute_iou(self, groundtruth_box, detection_box, isXywh=True):
        g_ymin, g_xmin, g_ymax, g_xmax = groundtruth_box
        d_ymin, d_xmin, d_ymax, d_xmax = detection_box
        if isXywh:
            g_ymax += g_ymin
            g_xmax += g_xmin
            d_ymax += d_ymin
            d_xmax += d_xmin
        xa = max(g_xmin, d_xmin)
        ya = max(g_ymin, d_ymin)
        xb = min(g_xmax, d_xmax)
        yb = min(g_ymax, d_ymax)

        intersection = max(0, xb - xa + 1) * max(0, yb - ya + 1)

        boxAArea = (g_xmax - g_xmin + 1) * (g_ymax - g_ymin + 1)
        boxBArea = (d_xmax - d_xmin + 1) * (d_ymax - d_ymin + 1)

        return intersection / float(boxAArea + boxBArea - intersection)

    def getConfusionMatrix(self):
        coco_eval = self.coco_eval
        categories = coco_eval.params.catIds
        confusion_matrix = np.zeros(shape=(len(categories) + 1, len(categories) + 1), dtype=np.int64)

        for imgId in coco_eval.params.imgIds:
            if imgId % 100 == 0 or imgId == len(coco_eval.params.imgIds)-1:
                print("Processed %d images" % (imgId))

            p = coco_eval.params
            gt = [_ for cId in p.catIds for _ in coco_eval._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in coco_eval._dts[imgId, cId] if _['score'] >= self.CONFIDENCE_THRESHOLD]
            g = [g['bbox'] for g in gt]
            d = [d['bbox'] for d in dt]
            matches = []
            for i in range(len(g)):
                for j in range(len(d)):
                    # print(f'{g[i]} {d[j]}')
                    iou = self._compute_iou(g[i], d[j], isXywh=True)
                    # print(iou)
                    if iou >= self.IOU_THRESHOLD:
                        matches.append([i, j, iou])
            matches = np.array(matches)

            if matches.shape[0] > 0:
                # Sort list of matches by descending IOU so we can remove duplicate detections
                # while keeping the highest IOU entry.
                matches = matches[matches[:, 2].argsort()[::-1][:len(matches)]]
                # Remove duplicate detections from the list.
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]

                # Sort the list again by descending IOU. Removing duplicates doesn't preserve
                # our previous sort.
                matches = matches[matches[:, 2].argsort()[::-1][:len(matches)]]

                # Remove duplicate ground truths from the list.
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]

            for i in range(len(g)):
                if matches.shape[0] > 0 and matches[matches[:, 0] == i].shape[0] == 1:
                    confusion_matrix[gt[i]['category_id'] -
                                     1][dt[int(matches[matches[:, 0] == i, 1][0])]['category_id'] - 1] += 1
                else:
                    confusion_matrix[gt[i]['category_id'] - 1][confusion_matrix.shape[1] - 1] += 1

            for i in range(len(d)):
                # if matches.shape[0] > 0 and matches[matches[:, 1] == i].shape[0] == 0:  # 逻辑错误，当没有匹配项时，FN应该加1，原先判断条件会跳过
                if matches.shape[0] == 0 or matches[matches[:, 1] == i].shape[0] == 0:
                    confusion_matrix[confusion_matrix.shape[0] - 1][dt[i]['category_id'] - 1] += 1

        return confusion_matrix

    def getMapMar(self):
        coco_eval = self.coco_eval
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize_per_category()
        # coco_eval.summarize()
        metric = coco_eval.stats[0]  # mAP 0.5-0.95
        return metric
