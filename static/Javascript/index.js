// JavaScript source code



    // Resets the form on refresh
    window.addEventListener('load', function () {
            var form = document.getElementById('myForm');
    form.reset();
        });
    window.addEventListener('DOMContentLoaded', function () {
            var inputField = document.querySelector('input');
    inputField.classList.add('blink-caret');
    inputField.focus();
        });
    // Get all the input fields in the form
    const inputFields = document.querySelectorAll('input');

        // Add event listener to each input field
        inputFields.forEach((field, index) => {
        field.addEventListener('keydown', function (event) {
            if (event.key === 'Enter') {
                event.preventDefault(); // Prevent form submission

                const nextIndex = index + 1;
                if (nextIndex < inputFields.length) {
                    inputFields[nextIndex].focus(); // Move to the next field
                } else {
                    // Reached the last field, submit the form
                    document.getElementById('myForm').submit();
                    console.log('Form submitted!')
                }
            }
        });
        });


//Form Validation
function validateForm() {
    // Perform form validation here
    var isValid = true;

    // Name validation
    var Name = document.getElementById("name").value;
    if (Name.trim() === "") {
        alert("Please enter your name");
        isValid = false;
    }

    // Gender validation
    var genderInputs = document.querySelectorAll('input[name="gender"]');
    var selectedGender = false;
    for (var i = 0; i < genderInputs.length; i++) {
        if (genderInputs[i].checked) {
            selectedGender = true;
            break;
        }
    }
    if (!selectedGender) {
        alert("Please select your gender");
        isValid = false;
    }

    // Age validation
    var age = document.getElementById("age").value;
    if (isNaN(age) || age < 20 || age > 90) {
        alert("Please enter a valid age between 20 and 90");
        isValid = false;
    }

    // Phone validation
    var phone = document.getElementById("phone").value;
    if (!/^\d{10}$/.test(phone)) {
        alert("Please enter a valid 10-digit phone number");
        isValid = false;
    }

    // Email validation
    var email = document.getElementById("email").value;
    if (!/^[\w-]+(\.[\w-]+)*@([\w-]+\.)+[a-zA-Z]{2,7}$/.test(email)) {
        alert("Please enter a valid email address");
        isValid = false;
    }

    if (!isValid) {
        // Form is invalid, proceed to the next page
        alert('Please fill in all required fields correctly.'); // Example of displaying an error message
        return false; // Prevent default form submission
    } 
}
