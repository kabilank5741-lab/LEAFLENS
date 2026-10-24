"""Train a lightweight plant-leaf classifier and export a browser/offline model.

Expected dataset layout (folder names become labels):
  dataset/train/Tomato___Early_blight/*.jpg
  dataset/train/Tomato___healthy/*.jpg
  dataset/validation/Tomato___Early_blight/*.jpg
  dataset/validation/Tomato___healthy/*.jpg

This script cannot invent a model without a labelled dataset. Training time varies.
"""
from pathlib import Path
import json, argparse
import tensorflow as tf

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
FRONTEND = PROJECT / 'frontend'

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',default=str(HERE/'dataset')); p.add_argument('--epochs',type=int,default=5); p.add_argument('--batch-size',type=int,default=16); p.add_argument('--export-tfjs',action='store_true'); args=p.parse_args()
    data=Path(args.data); train_dir=data/'train'; val_dir=data/'validation'
    if not train_dir.exists() or not val_dir.exists(): raise SystemExit('Dataset folders missing. Create dataset/train/<class_name>/ and dataset/validation/<class_name>/ with labelled leaf images.')
    train=tf.keras.utils.image_dataset_from_directory(train_dir,image_size=(224,224),batch_size=args.batch_size,label_mode='categorical')
    val=tf.keras.utils.image_dataset_from_directory(val_dir,image_size=(224,224),batch_size=args.batch_size,label_mode='categorical',shuffle=False)
    names=train.class_names
    if val.class_names != names: raise SystemExit('Train/validation class folders must match exactly and use the same names.')
    if len(names)<2: raise SystemExit('Need at least 2 labelled classes.')
    AUTOTUNE=tf.data.AUTOTUNE; train=train.prefetch(AUTOTUNE); val=val.prefetch(AUTOTUNE)
    base=tf.keras.applications.MobileNetV2(input_shape=(224,224,3),include_top=False,weights='imagenet')
    base.trainable=False
    inp=tf.keras.Input((224,224,3)); x=tf.keras.layers.Rescaling(1./127.5, offset=-1)(inp); x=base(x,training=False); x=tf.keras.layers.GlobalAveragePooling2D()(x); x=tf.keras.layers.Dropout(0.2)(x); out=tf.keras.layers.Dense(len(names),activation='softmax')(x)
    model=tf.keras.Model(inp,out); model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),loss='categorical_crossentropy',metrics=['accuracy'])
    model.fit(train,validation_data=val,epochs=max(1,args.epochs),callbacks=[tf.keras.callbacks.EarlyStopping(patience=2,restore_best_weights=True)])
    model_path=HERE/'model.keras'; labels_path=HERE/'class_names.json'; model.save(model_path); labels_path.write_text(json.dumps(names,indent=2),encoding='utf-8')
    print('Saved:',model_path); print('Labels:',labels_path); print('Validation metrics:',model.evaluate(val,verbose=0))
    if args.export_tfjs:
        try: import tensorflowjs as tfjs
        except ImportError: raise SystemExit('Install tensorflowjs from requirements-training.txt before --export-tfjs.')
        outdir=FRONTEND/'model'; outdir.mkdir(parents=True,exist_ok=True); tfjs.converters.save_keras_model(model,str(outdir)); (outdir/'class_names.json').write_text(json.dumps(names,indent=2),encoding='utf-8'); print('Browser model exported to',outdir)
    print('IMPORTANT: This is a baseline. Test on real field photos and get agricultural expert validation before deployment.')
if __name__=='__main__': main()
