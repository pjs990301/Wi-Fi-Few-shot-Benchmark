python main.py --learning supervised --dataset UT_HAR --model LeNet
python main.py --learning supervised --dataset UT_HAR --model ResNet50
python main.py --learning supervised --dataset UT_HAR --model RNN
python main.py --learning supervised --dataset UT_HAR --model LSTM
python main.py --learning supervised --dataset UT_HAR --model BiLSTM
python main.py --learning supervised --dataset UT_HAR --model ViT

python main.py --learning supervised --dataset ReWiS --model LeNet
python main.py --learning supervised --dataset ReWiS --model ResNet50
python main.py --learning supervised --dataset ReWiS --model RNN
python main.py --learning supervised --dataset ReWiS --model LSTM
python main.py --learning supervised --dataset ReWiS --model BiLSTM
python main.py --learning supervised --dataset ReWiS --model ViT

python main.py --learning supervised --dataset ReWiS --split T --model LeNet
python main.py --learning supervised --dataset ReWiS --split T --model ResNet50
python main.py --learning supervised --dataset ReWiS --split T --model RNN
python main.py --learning supervised --dataset ReWiS --split T --model LSTM
python main.py --learning supervised --dataset ReWiS --split T --model BiLSTM
python main.py --learning supervised --dataset ReWiS --split T --model ViT

# python main.py --learning few-shot --dataset ReWiS

# python main.py --learning few-shot --dataset ReWiS --model ViT