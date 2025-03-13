from django.shortcuts import render,redirect
from Face_Detection.detection import FaceRecognition
from .forms import *
from django.contrib import messages

faceRecognition = FaceRecognition()

def home(request):
    return render(request,'faceDetection/home.html')


def register(request):
    if request.method == "POST":
        form = RegistrationForm(request.POST, request.FILES)  # Include request.FILES
        if form.is_valid():
            form.save()
            print("IN HERE")
            messages.success(request, "Successfully registered")
            addFace(request.POST['face_id'])
            return redirect('home')  # Add return to properly redirect
        else:
            messages.error(request, "Account registration failed")
    else:
        form = RegistrationForm()

    return render(request, 'faceDetection/register.html', {'form': form})

def addFace(face_id):
    face_id = face_id
    faceRecognition.faceDetect(face_id)
    faceRecognition.trainFace()
    return redirect('/')

# def login(request):
#     face_id = faceRecognition.recognizeFace()
#     print(face_id)
#     return redirect('greeting' ,str(face_id))

def login(request):
    face_id = faceRecognition.recognizeFace()
    
    if face_id is None:  # No face recognized
        messages.error(request, "User Not Recognized")  # Show error message
        return render(request, 'faceDetection/user_not_recognized.html')  # Show error page

    print(face_id)
    return redirect('greeting', str(face_id))  # Redirect to greeting page if recognized


def Greeting(request, face_id):
    face_id = int(face_id)  # Convert to integer
    user = UserProfile.objects.filter(face_id=face_id).first()  # Get user safely

    if not user:
        return render(request, 'faceDetection/user_not_recognized.html')  # Show error page

    return render(request, 'faceDetection/greeting.html', {'user': user})

def logout(request):
    return redirect('home')