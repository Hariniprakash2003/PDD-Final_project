CREATE TABLE patient_details (
    patient_id INTEGER PRIMARY KEY DEFAULT floor(random() * 1000000)::integer,
    name VARCHAR(100) NOT NULL,
    age INTEGER NOT NULL,
    gender VARCHAR(10) NOT NULL,
    phone VARCHAR(20) NOT NULL,
    email VARCHAR(100) NOT NULL,
    result VARCHAR(10)
);
