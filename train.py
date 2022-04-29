import tensorflow as tf
from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, TensorBoard
from keras.losses import MeanSquaredError
from keras.models import load_model

import argparse
import datetime


from dataset_loader import *
from config import *
from model import auto_encoder
from losses import mean_gradient_error


print(tf.test.is_gpu_available())



def train(model_name, log_dir, learning_rate=1e-5, pretrained=PRETRAINED):
    model=None
    if pretrained:
        model=load_model(f"{MODEL_NAME}.h5")#, custom_objects={'mean_gradient_error': mean_gradient_error})
    else:
        model=auto_encoder()



    if not os.path.exists(MODEL_IMAGE_NAME):
        plot_model(
            model,
            to_file=MODEL_IMAGE_NAME,
            show_shapes=True,
            show_layer_names=True,
            show_layer_activations=True
        )


    train_high_res_set, train_low_res_set, validation_high_res_set, validation_low_res_set =load_data(
        TRAIN_HIGH_RES_DATA_FOLDER, 
        TRAIN_LOW_RES_DATA_FOLDER, 
        VALIDATION_HIGH_RES_DATA_FOLDER, 
        VALIDATION_LOW_RES_DATA_FOLDER
    )
    

    
    # model debug
    """
    cv2.imshow("train", train_high_res_set[0].numpy()/255.0)
    cv2.imshow("low res", train_low_res_set[0].numpy()/255.0)
    #print(model.predict(tf.expand_dims(train_set[0], axis=0)).shape)
    res=model.predict(tf.expand_dims(train_low_res_set[0], axis=0))[0]
    print(np.max(res), np.min(res))
    cv2.imshow("reconstructed", res/255.0)

    loss=mean_gradient_error(tf.expand_dims(train_high_res_set[0], axis=0), tf.expand_dims(res, axis=0))
    loss2=MeanSquaredError()(train_high_res_set[0], res).numpy()
    print("losses: ", loss, loss2)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    exit(0)
    """

    checkpoint=ModelCheckpoint(
        filepath=f"{model_name}.h5",
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        mode="min"
    )

    earlystopping=EarlyStopping(
        monitor="val_loss",
        patience=10,
        verbose=1,
        mode="min",
        restore_best_weights=True
    )

    reducelr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.1,
        patience=5,
        min_lr=1e-8,
        verbose=1
    )

    csv_logger=CSVLogger(
        f"{model_name}_history.csv",
        separator=","
    )

    now=datetime.datetime.now().strftime("%Y_%m_%d___%H_%M_%S")

    log_path=os.path.join(log_dir, now)
    
    tensorboard=TensorBoard(
        log_dir=log_path,
        histogram_freq=1
    )

    # hyperparameter logs

    # HP_DROPOUT=hp.HParam('dropout', hp.Discrete([0.4]))#hp.RealInterval(0.1, 0.2))
    # HP_LEARNING_RATE= hp.HParam('learning_rate', hp.Discrete([learning_rate])) #hp.Discrete([0.001, 0.0005, 0.0001]))
    # HP_OPTIMIZER=hp.HParam('optimizer', hp.Discrete(["adadelta"]))#hp.Discrete(['adam', 'sgd', 'rmsprop']))
    # HP_LOSS= "MeanSquaredError"
    # with tf.summary.create_file_writer(log_path).as_default():
    #     hp.hparams_config(
    #         hparams=[HP_DROPOUT,  HP_OPTIMIZER, HP_LEARNING_RATE],
    #         metrics=[hp.Metric(HP_LOSS, display_name='loss_function')],
    #     )

    # callbacks array
    callbacks=[
        checkpoint,
        earlystopping,
        csv_logger,
        tensorboard,
        reducelr
    ]


    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=MeanSquaredError(),
    )



    history=model.fit(
        train_low_res_set,
        train_high_res_set,
        validation_data=(
            validation_low_res_set,
            validation_high_res_set
        ),
        epochs=30,
        batch_size=BATCH_SIZE,
        shuffle=True,
        callbacks=callbacks
    )



def main(args):
    model_name=args.model_name
    log_dir=args.log_dir

    train(model_name, log_dir)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", "-mn", default=MODEL_NAME)
    parser.add_argument("--log_dir", "-ld", default=LOG_FOLDER)

    args = parser.parse_args()

    main(args)


