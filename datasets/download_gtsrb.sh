set -e
# Download dataset
wget http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip
wget http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_Images.zip
wget http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_GT.zip

# Unzip dataset
unzip GTSRB_Final_Training_Images.zip
unzip GTSRB_Final_Test_Images.zip
unzip GTSRB_Final_Test_GT.zip
cut -d ";" -f 1,8 < GT-final_test.csv > out
tail -n +2 out > image_labels
mv image_labels GTSRB/Final_Test/Images
cd GTSRB/Final_Test/Images/

while read  line || [[ -n "$line" ]]; do
    arr=(${line//;/ })
    mkdir -p ${arr[1]:0:-1}
    mv ${arr[0]} ${arr[1]:0:-1}
done < image_labels
