document.addEventListener("DOMContentLoaded", function () {
	const container = document.querySelector(".container");
	const sidebar = document.querySelector(".sidebar");

	const observer = new IntersectionObserver((entries) => {
		entries.forEach((entry) => {
			if (entry.isIntersecting) {
				entry.target.classList.add("visible");
			}
		});
	});

	observer.observe(container);
	observer.observe(sidebar);
});

document.addEventListener("DOMContentLoaded", function () {
	const dropArea = document.getElementById("drop-area");
	const fileInput = document.getElementById("file-input");
	const fileList = document.getElementById("file-list");
	const predictButton = document.getElementById("predict-button");
	const cancelButton = document.getElementById("cancel-button");
	const maskPlaceholder = document.getElementById("mask-placeholder");

	const updateFileList = (files) => {
		fileList.innerHTML = ""; // Clear previous file list
		const validFiles = Array.from(files).filter((file) =>
			file.name.endsWith(".nii")
		);
		if (validFiles.length === 0) {
			fileList.textContent = "No valid .nii files uploaded.";
			return;
		}
		validFiles.forEach((file) => {
			const fileItem = document.createElement("div");
			fileItem.textContent = file.name;
			fileList.appendChild(fileItem);
		});
		fileList.insertAdjacentHTML(
			"beforeend",
			`<p>Total files: ${validFiles.length}</p>`
		);
	};

	// Handle file selection via file input
	fileInput.addEventListener("change", (event) => {
		updateFileList(event.target.files);
	});

	// Handle drag-and-drop
	dropArea.addEventListener("dragover", (event) => {
		event.preventDefault();
		dropArea.style.backgroundColor = "#e9e9e9";
	});

	dropArea.addEventListener("dragleave", () => {
		dropArea.style.backgroundColor = "#f9f9f9";
	});

	dropArea.addEventListener("drop", (event) => {
		event.preventDefault();
		dropArea.style.backgroundColor = "#f9f9f9";
		const files = event.dataTransfer.files;
		updateFileList(files);
	});

	// Handle clicking the drop area to open the file input
	dropArea.addEventListener("click", () => {
		fileInput.click();
	});

	// Predict button functionality
	predictButton.addEventListener("click", () => {
		maskPlaceholder.textContent = "Processing... Please wait.";
	});

	// Cancel button functionality
	cancelButton.addEventListener("click", () => {
		maskPlaceholder.textContent = "Prediction cancelled.";
		fileInput.value = ""; // Reset the file input
		fileList.innerHTML = ""; // Clear file list
	});
});
