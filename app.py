from flask import Flask,render_template,request
import pandas as pd
import pickle

app=Flask(__name__)
data=pd.read_csv("C:\\Users\\BANAR\\Desktop\\AdvancedCarpriceprediction\\AdvanceCarpriceprediction\\Dataset\\Traindatas.csv")
scalar=pickle.load(open("C:\\Users\\BANAR\\Desktop\\AdvancedCarpriceprediction\\AdvanceCarpriceprediction\\Models\\scaler.pkl",'rb'))
encoder=pickle.load(open("C:\\Users\\BANAR\\Desktop\\AdvancedCarpriceprediction\\AdvanceCarpriceprediction\\Models\\encoder.pkl",'rb'))
model=pickle.load(open("C:\\Users\\BANAR\Desktop\\AdvancedCarpriceprediction\\AdvanceCarpriceprediction\Models\\RandomForestRegressors.pkl",'rb'))

list_of_new_columns=["Brand","Engine Type","Gear Box","Front Suspension","Rear Suspension","Front Brake Type","Body Type"]
list_of_numerical_columns=['ARAI Mileage(Km/L)',
                            'Engine Displacement (cc)',
                            'Fuel Tank Capacity',
                            'Length (mm)',
                            'Width (mm)',
                            'City Mileage(Km/L)',
                            'BHP',
                            'RPM',
                            'NM',
                            'NM_RPM']
list_of_catagorical_columns=['Varient',
                            'Brand',
                            'Engine Type',
                            'Rear Suspension',
                            'Front Suspension',
                            'Front Brake Type',
                            'Emission Norm Compliance',
                            'Height Adjustable Front Seat Belts',
                            'Door Ajar Warning',
                            'Adjustable Seats',
                            'Cup Holders-Front',
                            'Trunk Light',
                            'Power Steering',
                            'Rear Window Washer',
                            'Low Fuel Warning Light',
                            'Ventilated Seats',
                            'Glove Compartment',
                            'Automatic Headlamps',
                            'Child Safety Locks',
                            'Air Conditioner',
                            'Anti Lock Braking System',
                            'Driver Airbag',
                            'Body Type',
                            'Height Adjustable Driver Seat',
                            'Power Windows Front',
                            'Rear Seat Headrest',
                            'Passenger Side Rear View Mirror',
                            'Engine Check Warning',
                            'Adjustable Headlights',
                            'Rear Camera',
                            'Heater',
                            'Side Airbag-Front',
                            'Leather Steering Wheel',
                            'Accessory Power Outlet',
                            'Engine Immobilizer',
                            'Passenger Airbag',
                            'Audio System Remote Control',
                            'Dual Tone Dashboard',
                            'EBD',
                            'Electronic Stability Control',
                            'Power Windows Rear',
                            'Day & Night Rear View Mirror',
                            'Rear Seat Centre Arm Rest',
                            'Adjustable Steering',
                            'Crash Sensor',
                            'Voice Control',
                            'Apple CarPlay',
                            'Power Adjustable Exterior Rear View Mirror',
                            'Android Auto',
                            'USB & Auxiliary input',
                            'Speakers Rear',
                            'Model',
                            'Place',
                            'Drive Type',
                            'Bluetooth Connectivity',
                            'Halogen Headlamps',
                            'Multi-function Steering Wheel',
                            'Driving Experience Control Eco',
                            'ISOFIX Child Seat Mounts',
                            'Electric Folding Rear View Mirror',
                            'Smart Access Card Entry',
                            'Side Impact Beams',
                            'Rear AC Vents',
                            'Seat Lumbar Support',
                            'Gear Box',
                            'Option']
except_yes_catagory=['Varient',
                    'Brand',
                    'Engine Type',
                    'Rear Suspension',
                    'Front Suspension',
                    'Front Brake Type',
                    'Emission Norm Compliance',
                    'Body Type',
                    'Model',
                    'Place',
                    'Drive Type',
                    'Gear Box']
yes_catagory=['Height Adjustable Front Seat Belts',
            'Door Ajar Warning',
            'Adjustable Seats',
            'Cup Holders-Front',
            'Trunk Light',
            'Power Steering',
            'Rear Window Washer',
            'Low Fuel Warning Light',
            'Ventilated Seats',
            'Glove Compartment',
            'Automatic Headlamps',
            'Child Safety Locks',
            'Air Conditioner',
            'Anti Lock Braking System',
            'Driver Airbag',
            'Height Adjustable Driver Seat',
            'Power Windows Front',
            'Rear Seat Headrest',
            'Passenger Side Rear View Mirror',
            'Engine Check Warning',
            'Adjustable Headlights',
            'Rear Camera',
            'Heater',
            'Side Airbag-Front',
            'Leather Steering Wheel',
            'Accessory Power Outlet',
            'Engine Immobilizer',
            'Passenger Airbag',
            'Audio System Remote Control',
            'Dual Tone Dashboard',
            'EBD',
            'Electronic Stability Control',
            'Power Windows Rear',
            'Day & Night Rear View Mirror',
            'Rear Seat Centre Arm Rest',
            'Adjustable Steering',
            'Crash Sensor',
            'Voice Control',
            'Apple CarPlay',
            'Power Adjustable Exterior Rear View Mirror',
            'Android Auto',
            'USB & Auxiliary input',
            'Speakers Rear',
            'Bluetooth Connectivity',
            'Halogen Headlamps',
            'Multi-function Steering Wheel',
            'Driving Experience Control Eco',
            'ISOFIX Child Seat Mounts',
            'Electric Folding Rear View Mirror',
            'Smart Access Card Entry',
            'Side Impact Beams',
            'Rear AC Vents',
            'Seat Lumbar Support',
            'Option']
res=yes_catagory+except_yes_catagory
list_of_columns_to_keep=list_of_numerical_columns+list_of_catagorical_columns

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/prediction')
def render():
    return render_template("prediction.html",data=data,li=list_of_new_columns)

@app.route('/code')
def web():
    return render_template("code_webscrapping.html")

@app.route('/dataspliting')
def split():
    return render_template("code_dataspliting.html")

@app.route('/test')
def test():
    return render_template("code_test.html")

@app.route('/train')
def train():
    return render_template("code_train.html")

@app.route('/trainmodel')
def trainmodel():
    return render_template("code_train_model.html")

@app.route("/result",methods=["POST"])
def show_result():
    form_data=request.form
    temp={}
    for key in list_of_numerical_columns:
        temp["{}".format(key)]=float(form_data["{}".format(key)])
    for key in list_of_catagorical_columns:
        temp["{}".format(key)]=form_data["{}".format(key)]
    datas=pd.DataFrame(temp,index=[0])
    datas=datas[list_of_columns_to_keep]
    numerical_columns=[i for i in datas.columns if datas["{}".format(i)].dtype==float]
    scaled_value=scalar.transform(datas[numerical_columns])
    scaled_result=pd.DataFrame(scaled_value)
    for i,j in zip(list(range(0,scaled_result.shape[1])),numerical_columns):
        scaled_result.rename(columns={
            i:"{}".format(j)
        },inplace=True)
    transform_data=encoder.transform(datas[res])
    encoded_features=pd.DataFrame(transform_data)

    combined_data=pd.concat([scaled_result,encoded_features],axis=1)
    result_=(model.predict(combined_data)[0])

    return render_template("result.html",form_datas=datas,price=result_)

if __name__=="__main__":
    app.run(host="0.0.0.0",port=8080)