print("Starting..")
from flask import Flask,render_template,request,json,jsonify,session,redirect,send_file,url_for,flash
import os
from werkzeug.utils import secure_filename


import cv2
import numpy as np
from PIL import Image 

# import the necessary packages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from tensorflow.keras.models import load_model
import numpy as np
import cv2

IMAGE_SIZE = (224, 224, 3)

# CATEGORIES = ['C17', 'EF2000', 'F35', 'Vulcan', 'Mig31', 'Rafale', 'US2', 'V22', 'Tu160', 'RQ4', 'AG600', 'A400M', 'F15', 'C5', 'F18', 'Tornado', 'Su34', 'F16', 'F4', 'Su57', 'AV8B', 'MQ9', 'JAS39', 'Mirage2000', 'B52', 'U2', 'XB70', 'C130', 'YF23', 'F22', 'F117', 'E2', 'B1', 'A10', 'B2', 'J20', 'Be200', 'F14', 'Tu95', 'SR71'] 
CATEGORIES = ['A10','A400M', 'AG600','AV8B', 'B1', 'B2', 'B52', 'Be200', 'C130', 'C17', 'C5', 'E2', 'EF2000', 'F117', 'F14', 'F15', 'F16', 'F18', 'F22', 'F35', 'F4', 'J20', 'JAS39', 'Mig31', 'Mirage2000', 'MQ9', 'Rafale', 'RQ4','SR71','Su34','Su57','Tornado','Tu160','Tu95','U2','US2','V22','Vulcan','XB70','YF23'] 


SOLUTIONS = {
    'C17': """
    The McDonnell Douglas/Boeing C-17 Globemaster III is a large military transport aircraft that was developed for the United States Air Force (USAF) from the 1980s to the early 1990s by McDonnell Douglas. The C-17 carries forward the name of two previous piston-engined military cargo aircraft, the Douglas C-74 Globemaster and the Douglas C-124 Globemaster II.
    """, 
    'EF2000': """
    The Eurofighter Typhoon is a European multinational twin-engine, canard delta wing, multirole fighter. The Typhoon was designed originally as an air superiority fighter[5] and is manufactured by a consortium of Airbus, BAE Systems and Leonardo that conducts the majority of the project through a joint holding company, Eurofighter Jagdflugzeug GmbH. The NATO Eurofighter and Tornado Management Agency, representing the UK, Germany, Italy and Spain, manages the project and is the prime customer.
    """, 
    'F35': """
    The Lockheed Martin F-35 Lightning II is an American family of single-seat, single-engine, all-weather stealth multirole combat aircraft that is intended to perform both air superiority and strike missions. It is also able to provide electronic warfare and intelligence, surveillance, and reconnaissance capabilities. Lockheed Martin is the prime F-35 contractor, with principal partners Northrop Grumman and BAE Systems. The aircraft has three main variants: the conventional takeoff and landing (CTOL) F-35A, the short take-off and vertical-landing (STOVL) F-35B, and the carrier-based (CV/CATOBAR) F-35C.
    """, 
    'Vulcan': """
    The Avro Vulcan (later Hawker Siddeley Vulcan from July 1963) is a jet-powered, tailless, delta-wing, high-altitude, strategic bomber, which was operated by the Royal Air Force (RAF) from 1956 until 1984.
    """, 
    'Mig31': """
    MiG-31BM has multi-role capability as is capable of using anti-radar, air to ship and air to ground missiles. It has some of avionics unified with MiG-29SMT and has refueling probe. MiG-31BM broke world record while spending seven hours and four minutes in the air while covering the distance of 8,000 km (5,000 mi).
    """, 
    'Rafale': """
    The aircraft is available in three main variants: Rafale C single-seat land-based version, Rafale B twin-seat land-based version, and Rafale M single-seat carrier-based version. Introduced in 2001, the Rafale is being produced for both the French Air Force and for carrier-based operations in the French Navy.
    """, 
    'US2': """
    The ShinMaywa US-2 is a large Japanese short takeoff and landing amphibious aircraft developed and manufactured by seaplane specialist ShinMaywa (formerly Shin Meiwa). It was developed from the earlier Shin Meiwa US-1A seaplane, which was introduced during the 1970s.
    """, 
    'V22': """
    The Bell Boeing V-22 Osprey is an American multi-mission, tiltrotor military aircraft with both vertical takeoff and landing (VTOL) and short takeoff and landing (STOL) capabilities.
    """, 
    'Tu160': """
    The Tu-160 is a variable-geometry wing aircraft. The aircraft employs a fly-by-wire control system with a blended wing profile, and full-span slats are used on the leading edges, with double-slotted flaps on the trailing edges and cruciform tail.
    """, 
    'RQ4': """
    The RQ-4 Global Hawk is a high-altitude, long-endurance, remotely piloted aircraft with an integrated sensor suite that provides global all-weather, day or night intelligence, surveillance and reconnaissance (ISR) capability.
    """, 
    'AG600': """
    t has an overall length of 36.9m, height of 12.1m and wingspan of 38.8m. The aircraft can take off and land from 1,500m-long, 200m-wide and 2.5m-deep water bodies. It has the capacity to collect 12t of water in 20 seconds and can carry up to 370t of water on a single tank of fuel.
    """, 
    'A400M': """
    he Airbus A400M Atlas is a European four-engine turboprop military transport aircraft. It was designed by Airbus Military (now Airbus Defence and Space) as a tactical airlifter with strategic capabilities to replace older transport aircraft, such as the Transall C-160 and the Lockheed C-130 Hercules.
    """, 
    'F15': """
    The F-15 Eagle is an all-weather, extremely maneuverable, tactical fighter designed to permit the Air Force to gain and maintain air supremacy over the battlefield. The Eagle's air superiority is achieved through a mixture of unprecedented maneuverability and acceleration, range, weapons and avionics.
    """, 
    'C5': """
    The C-5 is a large, high-wing cargo aircraft with a distinctive high T-tail fin (vertical) stabilizer, with four TF39 turbofan engines mounted on pylons beneath wings that are swept 25°. (The C-5M uses newer GE CF6 engines.)
    """, 
    'F18': """
    The McDonnell Douglas F/A-18 Hornet is an all-weather, twin-engine, carrier-capable, multirole combat aircraft, designed as both a fighter and attack aircraft (hence the F/A designation).
    """, 
    'Tornado': """
    The Tornado ADV was outfitted with beyond visual range AAMs such as the Skyflash and AIM-120 AMRAAM missiles. The Tornado is armed with two 27 mm (1.063 in) Mauser BK-27 revolver cannon internally mounted underneath the fuselage; the Tornado ADV was only armed with one cannon.
    """, 
    'Su34': """
    The Su-34 is powered by a pair of Saturn AL-31FM1 turbofan engines, the same engines used on the Su-27SM, giving the aircraft a maximum speed of Mach 1.8+ when fully loaded. Although slower than the standard Su-27, the Su-34 can still handle high G-loads and perform aerobatic maneuvers.
    """, 
    'F16': """
    The F-16 Fighting Falcon is a compact, multi-role fighter aircraft. It is highly maneuverable and has proven itself in air-to-air combat and air-to-surface attack. It provides a relatively low-cost, high-performance weapon system for the United States and allied nations.
    """, 
    'F4': """
    Innovations in the F-4 included an advanced pulse-Doppler radar and extensive use of titanium in its airframe. Despite imposing dimensions and a maximum takeoff weight of over 60,000 lb (27,000 kg), the F-4 has a top speed of Mach 2.23 and an initial climb rate of over 41,000 ft/min (210 m/s).
    """, 
    'Su57': """
    The Su-57 is the first aircraft in Russian military service designed with stealth technology and is intended to be the basis for a family of stealth combat aircraft.
    """, 
    'AV8B': """
    The aircraft's internal fuel capacity is 7,500 lb (3,400 kg), up 50% compared to its predecessor. Fuel capacity can be carried in hardpoint-compatible external drop tanks, which give the aircraft a maximum ferry range of 2,100 mi (3,300 km) and a combat radius of 300 mi (556 km)
    """, 
    'MQ9': """
    The MQ-9 is a larger, heavier, and more capable aircraft than the earlier General Atomics MQ-1 Predator; it can be controlled by the same ground systems used to control MQ-1s. The Reaper has a 950-shaft-horsepower (712 kW) turboprop engine (compared to the Predator's 115 hp (86 kW) piston engine).
    """, 
    'JAS39': """
    The Saab JAS 39 Gripen (IPA: [ˈɡrǐːpɛn]; English: griffin) is a light single-engine multirole fighter aircraft manufactured by the Swedish aerospace and defense company Saab AB. The Gripen has a delta wing and canard configuration with relaxed stability design and fly-by-wire flight controls.
    """, 
    'Mirage2000': """
    The Dassault Mirage 2000 is a French multirole, single-engine, fourth-generation jet fighter manufactured by Dassault Aviation. It was designed in the late 1970s as a lightweight fighter to replace the Mirage III for the French Air Force (Armée de l'air).
    """, 
    'B52': """
    The B-52 has a wingspan of 185 feet (56 metres) and a length of 160 feet 10.9 inches (49 metres). It is powered by eight jet engines mounted under the wings in four twin pods. The plane's maximum speed at 55,000 feet (17,000 metres) is Mach 0.9 (595 miles per hour, or 960 km/hr)
    """, 
    'U2': """
    The U-2 aircraft, built of aluminum and limited to subsonic flight, can cruise for many hours above 70,000 feet (21,000 meters) with a payload weighing 3,000 pounds (1,350 kg). Its exact operational specifications are secret.
    """, 
    'XB70': """
    XB-70 No. 1 surpassed Mach 3 on 14 October 1965 by reaching Mach 3.02 at 70,000 ft (21,000 m). The first aircraft was found to suffer from weaknesses in the honeycomb panels, primarily due to inexperience with fabrication and quality control of this new material.

    """, 
    'C130': """
    C-130H: 23,000 feet (7,077 meters) with 42,000 pounds (19,090 kilograms) payload. Maximum Load: C-130E/H/J: 6 pallets or 72 litters or 16 CDS bundles or 90 combat troops or 64 paratroopers, or a combination of any of these up to the cargo compartment capacity or maximum allowable weight.
    """, 
    'YF23': """
    The Northrop/McDonnell Douglas YF-23 is an American single-seat, twin-engine stealth fighter aircraft technology demonstrator designed for the United States Air Force (USAF). The design was a finalist in the USAF's Advanced Tactical Fighter (ATF) competition, battling the Lockheed YF-22 for a production contract.
    """, 
    'F22': """
    The Northrop/McDonnell Douglas YF-23 is an American single-seat, twin-engine stealth fighter aircraft technology demonstrator designed for the United States Air Force (USAF). The design was a finalist in the USAF's Advanced Tactical Fighter (ATF) competition, battling the Lockheed YF-22 for a production contract.
    """, 
    'F117': """
    The Lockheed F-117 Nighthawk is a retired American single-seat, twin-engine stealth attack aircraft developed by Lockheed's secretive Skunk Works division and operated by the United States Air Force (USAF). It was the first operational aircraft to be designed with stealth technology.
    """, 
    'E2': """
    The E2 features a closed loop fly-by-wire control which reduces weight, increases fuel efficiency, enhances control and increases safety by full envelope protection in all flight phases compared to the first E-Jet.
    """, 
    'B1': """
    The B-1B is 147 feet (44.8 metres) long, and, when fully extended, its wings span about 137 feet (42 metres). The plane's four General Electric turbofan engines can accelerate it past the speed of sound at its operating ceiling of 40,000 feet (12,000 metres), but its normal cruising speed is subsonic.
    """, 
    'A10': """
    The Fairchild Republic A-10 Thunderbolt II is a single-seat, twin-turbofan, straight-wing, subsonic attack aircraft developed by Fairchild Republic for the United States Air Force (USAF). In service since 1976, it is named for the Republic P-47 Thunderbolt, a World War II-era fighter-bomber effective at attacking ground targets, but commonly referred to as the "Warthog" or "Hog". The A-10 was designed to provide close air support (CAS) to friendly ground troops by attacking armored vehicles, tanks, and other enemy ground forces; it is the only production-built aircraft designed solely for CAS to have served with the U.S. Air Force.[5] Its secondary mission is to direct other aircraft in attacks on ground targets, a role called forward air controller-airborne; aircraft used primarily in this role are designated OA-10.

The A-10 was intended to improve on the performance and firepower of the Douglas A-1 Skyraider. Its airframe was designed for durability, with measures such as 1,200 pounds (540 kg) of titanium armor to protect the cockpit and aircraft systems, enabling it to absorb damage and continue flying. Its ability to take off and land from relatively short runways permits operation from airstrips close to the front lines, and its simple design enables maintenance with minimal facilities.
    """, 
    'B2': """
    The bomber can drop conventional and thermonuclear weapons, such as up to eighty 500-pound class (230 kg) Mk 82 JDAM GPS-guided bombs, or sixteen 2,400-pound (1,100 kg) B83 nuclear bombs. The B-2 is the only acknowledged aircraft that can carry large air-to-surface standoff weapons in a stealth configuration.
    """, 
    'J20': """
    The J-20 is designed as an air superiority fighter with precision strike capability. The aircraft has three variants: the initial production model J-20A, the thrust-vectoring J-20B, and twin-seat aircraft teaming capable J-20S.
    """, 
    'Be200': """
    The Beriev Be-200 Altair is a multipurpose amphibious aircraft designed by the Beriev Aircraft Company and manufactured by Irkut. Marketed as being designed for fire fighting, search and rescue, maritime patrol, cargo, and passenger transportation, it has a capacity of 12 tonnes of water, or up to 72 passengers.
    """, 
    'F14': """
    The Grumman F-14 Tomcat is an American carrier-capable supersonic, twin-engine, two-seat, twin-tail, variable-sweep wing fighter aircraft. The Tomcat was developed for the United States Navy's Naval Fighter Experimental (VFX) program after the collapse of the General Dynamics-Grumman F-111B project.
    """, 
    'Tu95': """
    The Tu-95 is one of the loudest military aircraft, particularly because the tips of the propeller blades move faster than the speed of sound. Its distinctive swept-back wings are set at an angle of 35°. The Tu-95 is the only propeller-driven aircraft with swept wings that has been built in large numbers.
    """, 
    'SR71': """
    The SR-71 was powered by two Pratt & Whitney J58 (company designation JT11D-20) axial-flow turbojet engines. The J58 was a considerable innovation of the era, capable of producing a static thrust of 32,500 lbf (145 kN). The engine was most efficient around Mach 3.2, the Blackbird's typical cruising speed.
    """, 
    
}


model = load_model("resnet_101_v2_model.h5")

def model_warmup():
    dummy_image = []
    for i in range(224):
        dummy_image.append([[0]*3]*224)
    image = np.array(dummy_image)
    # print(image.shape)
    image = np.expand_dims(image, axis=0)
    pred = model.predict(image)
    # print(pred)



def predict(imgpath):
    img = cv2.imread(imgpath)
    img = cv2.resize(img, (IMAGE_SIZE[0], IMAGE_SIZE[1]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_rotated_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    img_rotated_180 = cv2.rotate(img, cv2.ROTATE_180)
    img_rotated_270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    img_flip_ver = cv2.flip(img, 0)
    img_flip_hor = cv2.flip(img, 1)

    images = []
    images.append(img)
    images.append(img_rotated_90)
    images.append(img_rotated_180)
    images.append(img_rotated_270)
    images.append(img_flip_ver)
    images.append(img_flip_hor)

    images = np.array(images)
    images = images.astype(np.float32)
    images /= 255

    op = []
    # make predictions on the input image
    for im in images:
        image = np.array(im)
        image = np.expand_dims(image, axis=0)
        pred = model.predict(image)
        pred = pred.argmax(axis=1)[0]
        op.append(pred)
        print("Pred:", pred, CATEGORIES[pred])

    op = np.array(op)
    print("Final Output:", CATEGORIES[np.bincount(op).argmax()])
    return CATEGORIES[np.bincount(op).argmax()]


model_warmup()



app=Flask(__name__)
app.secret_key="secure"
app.config['UPLOAD_FOLDER'] = str(os.getcwd())+'/static/uploads'

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/',methods=["post","get"])
def first_page():
    if request.method=="POST":
        global image_name,image_data

        file = request.files['file']
        if file.filename == '':
            flash('No image selected for uploading')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            op = predict('static/uploads/'+filename)
            solution = SOLUTIONS[op]
            return render_template("data_page.html",
                           filename=filename, result = op, solution = solution.split("\n"))
        else:
            flash('Allowed image types are -> png, jpg, jpeg, gif')
            return redirect(request.url)

    else:
        return render_template("form_page.html")


@app.route('/display/<filename>')
def display_image(filename):
	#print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='uploads/' + filename), code=301)

app.run(debug=True, host="0.0.0.0")
