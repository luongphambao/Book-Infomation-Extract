const dropArea = document.querySelector(".drag-area");
const dragText = document.querySelector(".header");

let button = dropArea.querySelector(".button");
let input = dropArea.querySelector("input");

let file;

button.onclick = () => {
	input.click();
};

function loading_on() {
	document.querySelector(".overlay").style.display = "block";
}

function loading_off() {
	document.querySelector(".overlay").style.display = "none";
}

// when browse
input.addEventListener("change", function () {
	file = this.files[0];
	dropArea.classList.add("active");
	displayFile();
});

// when file is inside drag area
dropArea.addEventListener("dragover", (event) => {
	event.preventDefault();
	dropArea.classList.add("active");
	dragText.textContent = "Release to Upload";
	// console.log('File is inside the drag area');
});

// when file leave the drag area
dropArea.addEventListener("dragleave", () => {
	dropArea.classList.remove("active");
	// console.log('File left the drag area');
	dragText.textContent = "Drag & Drop";
});

// when file is dropped
dropArea.addEventListener("drop", (event) => {
	event.preventDefault();
	// console.log('File is dropped in drag area');

	file = event.dataTransfer.files[0]; // grab single file even of user selects multiple files
	// console.log(file);
	displayFile();
	// window.alert(file && file['type'].split('/')[0] === 'image');
});

const testClick = () => {
	console.log("Button clicked!")
}

document.querySelector("form").addEventListener("submit", function (e) {
	e.preventDefault();
	if (file == null) {
		alert("Please upload your image!");
	} else {
		loading_on();
		const formData = new FormData();
		formData.append("file", file);
		const xhr = new XMLHttpRequest();

		xhr.onload = function () {
			// if (xhr.readyState == XMLHttpRequest.DONE && xhr.status == 200) {
			// window.alert(xhr.status)
			// loading_off();
			// const data = JSON.parse(xhr.responseText).data;
			// const update =  new Date();
			// document.querySelector('.extracted__img').innerHTML = `<img src="/static/src/0.jpg?v=${update.getTime()}" />`; // To update avoid using image from cache
			// document.querySelector('.info__title').innerHTML = `Title: ${data[1]}`;
			// }
			// else if (xhr.status == 404){
			//   const data = JSON.parse(xhr.responseText).data;
			//   window.alert(data)
			//   loading_off()
			// }

			setTimeout(() => {
				loading_off();
			}, 1000);

			// pseudo extracting process

			const data = JSON.parse(xhr.responseText);
			console.log(data);

			if (data.status == "OK") {
				const update = new Date();
				document.querySelector(
					".info__title"
				).innerHTML = `Title: ${data.title}`;
				document.querySelector(
					".info__sub_title"
				).innerHTML = `Sub title: ${data.sub_title}`;
				document.querySelector(
					".info__author"
				).innerHTML = `Author: ${data.author}`;
				document.querySelector(
					".info__date"
				).innerHTML = `Date of publication: ${data.date}`;
				document.querySelector(
					".info__others"
				).innerHTML = `Others: ${data.others}`;

				document.querySelector(
					".extracted__img"
				).innerHTML = `<img src="/static/src/0.jpg?v=${update.getTime()}" />`; // To update avoid using image from cache
			} else {
				window.alert(data.status);
			}
		};

		let URL = "/extract";
		xhr.open("POST", URL);
		xhr.send(formData);
	}
});

function displayFile() {
	let fileType = file.type;
	// console.log(fileType);

	let validExtensions = ["image/jpeg", "image/jpg", "image/png"];

	if (validExtensions.includes(fileType)) {
		// console.log('This is an image file');
		let fileReader = new FileReader();

		fileReader.onload = () => {
			let fileURL = fileReader.result;
			// console.log(fileURL);
			let imgTag = `<img src="${fileURL}" alt="">`;
			dropArea.innerHTML = imgTag;
		};
		fileReader.readAsDataURL(file);
	} else {
		alert("This file is not supported!");
		dropArea.classList.remove("active");
	}
}
