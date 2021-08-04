import os
import logging
import cv2
from new_timer import AutoTimer


class IR_MTCNN(object):
    def __init__(self,
                 p_net_model=tuple(),
                 r_net_model=tuple(),
                 o_net_model=tuple()):
        """
        Args
        ----
        p_net_model:
        
        r_net_model:
        
        o_net_model:

        Attribute
        ---------
        p_net: P network.
        
        r_net: R network.
        
        o_net: O network.
        """
        self.__p_net_xml = p_net_model[0]
        self.__p_net_bin = p_net_model[1]
        self.__r_net_xml = r_net_model[0]
        self.__r_net_bin = r_net_model[1]
        self.__o_net_xml = o_net_model[0]
        self.__o_net_bin = o_net_model[1]

        self.__p_net = self.__load_network(self.__p_net_xml, self.__p_net_bin)
        self.__r_net = self.__load_network(self.__r_net_xml, self.__r_net_bin)
        self.__o_net = self.__load_network(self.__o_net_xml, self.__o_net_bin)


    def __load_network(self, model_xml=str(), model_bin=str()):
        """
        Args
        ----
        model_xml: IR xml format file.
        
        model_bin: IR bin format file.

        Return
        ------
        network: IR network.
        """

        if not os.path.exists(model_xml) or not os.path.exists(model_bin):
            logging.critical("{} or {} not existed !".format(model_xml, model_bin))
            raise FileNotFoundError

        network = cv2.dnn.readNetFromModelOptimizer(model_xml, model_bin)
        network.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
        network.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

        return network

    
    def P_net_predict(self, image):
        """
        Args
        ----
        image: Input RGB image.

        Return
        ------
        result: Prediction result.
        """
        self.__p_net.setInput(image)
        result = self.__p_net.forward()

        return result


    def R_net_predict(self, image):
        """
        Args
        ----
        image: Input RGB image.

        Return
        ------
        result: Prediction result.
        """
        self.__r_net.setInput(image)
        result = self.__r_net.forward()

        return result


    def O_net_predict(self, image):
        """
        Args
        ----
        image: Input RGB image.

        Return
        ------
        result: Prediction result.
        """
        self.__o_net.setInput(image)
        result = self.__o_net.forward()

        return result


if __name__ == "__main__":
    FORMAT = '%(asctime)s [%(levelname)s] %(message)s'
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    logging.basicConfig(level=logging.DEBUG, format=FORMAT, datefmt=DATE_FORMAT)


    image = cv2.imread("ivan_drawn.jpg")
    P_blob = cv2.dnn.blobFromImage(image, 1.0, (640, 480), swapRB=True)
    R_blob = cv2.dnn.blobFromImage(image, 1.0, (24, 24), swapRB=True)
    O_blob = cv2.dnn.blobFromImage(image, 1.0, (48, 48), swapRB=True)

    IR_mtcnn = IR_MTCNN(p_net_model=("mtcnn/models/pnet.xml", "mtcnn/models/pnet.bin"),
                        r_net_model=("mtcnn/models/rnet.xml", "mtcnn/models/rnet.bin"),
                        o_net_model=("mtcnn/models/onet.xml", "mtcnn/models/onet.bin"))
    
    for i in range(10):
        with AutoTimer("IR_MTCNN", 4):
            result = IR_mtcnn.P_net_predict(P_blob)
            result = IR_mtcnn.R_net_predict(R_blob)
            result = IR_mtcnn.O_net_predict(O_blob)
            # print(result)