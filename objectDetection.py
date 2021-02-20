
from __future__ import print_function
import sys
import os
import cv2
import time
from openvino.inference_engine import IENetwork, IECore
model ="model/person-vehicle-bike-detection-2000.xml"

device ="CPU"
input_stream = "video/cars.mp4"
labels ="model/labels.txt"
threshold= 0.6
is_async_mode = False # synchronous if False
def main():
    model_xml = model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    ie = IECore()
    net = IENetwork(model=model_xml, weights=model_bin)
    img_info_input_blob = None
    feed_dict = {}
    for blob_name in net.inputs:
        if len(net.inputs[blob_name].shape) == 4:
            input_blob = blob_name
        elif len(net.inputs[blob_name].shape) == 2:
            img_info_input_blob = blob_name
        else:
            raise RuntimeError("Unsupported {}D input layer '{}'. Only 2D and 4D input layers are supported"
                               .format(len(net.inputs[blob_name].shape), blob_name))
    assert len(net.outputs) == 1, "Demo supports only single output topologies"
    out_blob = next(iter(net.outputs))
    exec_net = ie.load_network(network=net, num_requests=2, device_name=device)
    # Read and pre-process input image
    n, c, h, w = net.inputs[input_blob].shape
    if img_info_input_blob:
        feed_dict[img_info_input_blob] = [h, w, 1]
    cap = cv2.VideoCapture(input_stream)
    assert cap.isOpened(), "Can't open " + input_stream
    with open(labels, 'r') as f:
        labels_map = [x.strip() for x in f]
    cur_request_id = 0
    next_request_id = 1
    
    
    if is_async_mode:
        ret, frame = cap.read()
        frame_h, frame_w = frame.shape[:2]
    
    while cap.isOpened():
        if is_async_mode:
            ret, next_frame = cap.read()
        else:
            ret, frame = cap.read()
            if ret:
                frame_h, frame_w = frame.shape[:2]
        if not ret:
            break  # abandons the last frame in case of async_mode
        inf_start = time.time()
        if is_async_mode:
            in_frame = cv2.resize(next_frame, (w, h))
            in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            in_frame = in_frame.reshape((n, c, h, w))
            feed_dict[input_blob] = in_frame
            exec_net.start_async(request_id=next_request_id, inputs=feed_dict)
        else:
            in_frame = cv2.resize(frame, (w, h))
            in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            in_frame = in_frame.reshape((n, c, h, w))
            feed_dict[input_blob] = in_frame
            exec_net.start_async(request_id=cur_request_id, inputs=feed_dict)
        if exec_net.requests[cur_request_id].wait(-1) == 0:
            inf_end = time.time()
            det_time = inf_end - inf_start

            # Parse detection results of the current request
            res = exec_net.requests[cur_request_id].outputs[out_blob]
            for obj in res[0][0]:
                # Draw only objects when probability more than specified threshold
                if obj[2] > threshold:
                    xmin = int(obj[3] * frame_w)
                    ymin = int(obj[4] * frame_h)
                    xmax = int(obj[5] * frame_w)
                    ymax = int(obj[6] * frame_h)
                    class_id = int(obj[1])
                    # Draw box and label\class_id
                    color = (min(class_id * 12.5, 255), min(class_id * 7, 255), min(class_id * 5, 255))
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,255,255), 2)
                    det_label = labels_map[class_id] if labels_map else str(class_id)
                    
                    cv2.putText(frame, det_label, (xmin, ymin - 7),
                                cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,255,255), 1)
            
            
            # Draw performance stats
            inf_time_message = "Fps: N\A " if is_async_mode else \
                "Fps: {}".format(int(det_time * 1000)) 
            cv2.rectangle(frame, (10,20),(130,55),(0,0,0),-1)

            cv2.putText(frame, inf_time_message, (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255,255), 2)
        
        cv2.imshow("Detection Results", frame)
        

        if is_async_mode:
            cur_request_id, next_request_id = next_request_id, cur_request_id
            frame = next_frame
            frame_h, frame_w = frame.shape[:2]

        key = cv2.waitKey(1)
        if key == 27:
            break
        
    cv2.destroyAllWindows()


if __name__ == '__main__':
    sys.exit(main() or 0)
