import csv
import copy

import cv2 as cv
import mediapipe as mp

from model import KeyPointClassifier
from app import calc_landmark_list, pre_process_landmark, get_args

from jokenpo import whoWins
import random
from time import sleep


def main():
    # Tratando os argumentos recebidos na chamada da linha da execução
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    # Configurando a captura da câmera
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Importando o modelo de hand-tracking
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    # Classificador de Pontos-chave (Formatos de Mão / [Pedra, Papel, Tesoura e OK])
    keypoint_classifier = KeyPointClassifier()

    # Importando o nome das classes processadas pelo classificador
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]

    while True:
        # Tecla de exceção da execução do loop de código
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break

        # Instanciando a captura de câmera
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)

        # Instanciando a detecção de mãos via MediaPipe
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        # Obtendo dados do tracking
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                # Cálculo das landmarks dentro do frame
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Normalização das coordenadas
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)

                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)

                debug_image = draw_info(debug_image, keypoint_classifier_labels[hand_sign_id])

        debug_image = draw_info(debug_image)
        cv.imshow('RPS Game',debug_image)

    cap.release()
    cv.destroyAllWindows()


def draw_info(image, pick=None):
    # Função para escrever informações na tela de acordo com as detecções
    global jogou, fim, jogada, cpu
    if fim == True:
        # Condição para fim da jogada
        cv.putText(image, 'Computador jogou: ' + cpu, (100, 150),
                   cv.FONT_HERSHEY_SIMPLEX,
                   1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, 'Computador jogou: ' + cpu, (100, 150),
                   cv.FONT_HERSHEY_SIMPLEX,
                   1.0, (255, 255, 255), 2, cv.LINE_AA)

        cv.putText(image, jogada, (200, 200),
                   cv.FONT_HERSHEY_SIMPLEX,
                   1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, jogada, (200, 200),
                   cv.FONT_HERSHEY_SIMPLEX,
                   1.0, (255, 255, 255), 2, cv.LINE_AA)

        if (pick == 'Tope') and len(janela_da_jogada) > 4:  # Condição para jogar novamente
            reset()  # Reiniciando as varíaveis de histórico
            sleep(3)
            return image

        return image

    elif jogou == True:
        # Jogada do Computador após jogada do usuário ser detectada
        cpu = random.choice(['Pedra', 'Papel', 'Tesoura'])
        jogada = whoWins(janela_da_jogada[-1], cpu)

        cv.putText(image, jogada, (200, 200), cv.FONT_HERSHEY_SIMPLEX,
                   1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, jogada, (200, 200), cv.FONT_HERSHEY_SIMPLEX,
                   1.0, (255, 255, 255), 2, cv.LINE_AA)
        fim = True  # Estabelecendo condição para fim da rodada

        return image

    elif len(janela_da_jogada) > 4:
        # Quantidade de detecções para contar como uma jogada
        cv.putText(image, 'Jogou: ' + str(janela_da_jogada[-1]), (10, 70),
                   cv.FONT_HERSHEY_SIMPLEX,
                   1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, 'Jogou: ' + str(janela_da_jogada[-1]), (10, 70),
                   cv.FONT_HERSHEY_SIMPLEX,
                   1.0, (255, 255, 255), 2, cv.LINE_AA)
        jogou = True  # Estabelecendo condição para o computador jogar

        return image

    elif (pick == None) or pick == "":
        # Não plotar nada caso não seja reconhecido a jogada
        cv.putText(image, "pick:", (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX,
                   1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "pick:", (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX,
                   1.0, (255, 255, 255), 2, cv.LINE_AA)

        return image
    cv.putText(image, "pick:" + pick, (10, 30),
               cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "pick:" + pick, (10, 30),
               cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)
    janela_da_jogada.append(pick)

    return image


def reset():
    # Função para voltar as variavéis de parada a condição inicial
    global janela_da_jogada
    janela_da_jogada = []
    global jogou, fim
    jogou = False
    fim = False


if __name__ == '__main__':
    janela_da_jogada = []
    jogou = False
    fim = False
    main()