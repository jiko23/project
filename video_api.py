import requests as res
import urllib.request

#example link: "https://static.videezy.com/system/resources/previews/000/046/915/original/Zeabra1.mp4"
video_link = input("Kindly enter the link of the video: ")

saving_address = r"E:\prog\canada\video" + ".mp4"

try:
	chunk_size = 256

	response = res.get(video_link, stream = True)

	with open(saving_address, "wb") as f:

		for chunk in response.iter_content(chunk_size = chunk_size):

			f.write(chunk)

		#f.write(response.data)

except:

	urllib.request.urlretrieve(video_link, saving_address)