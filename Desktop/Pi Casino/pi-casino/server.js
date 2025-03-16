require("dotenv").config();
const express = require("express");
const cors = require("cors");
const bodyParser = require("body-parser");
import { Payment } from "pi-backend";
import { Payment } from "pi-backend";

console.log(Payment); // Sollte eine Funktion oder Klasse ausgeben



const app = express();
app.use(cors());
app.use(bodyParser.json());

const PORT = process.env.PORT || 5000;

// Pi API-Keys (ersetze mit deinen eigenen)
const PI_API_KEY = process.env.PI_API_KEY;

// Starte Pi-Zahlungssystem
const piPayment = new Payment(PI_API_KEY);

// Route für Pi-Einzahlung
app.post("/deposit", async (req, res) => {
    const { userId, amount } = req.body;

    try {
        // Erstelle eine Pi-Transaktion
        const payment = await piPayment.createPayment(userId, amount);
        res.json({ success: true, payment });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

// Starte Server
app.listen(PORT, () => {
    console.log(`🚀 Pi Casino Backend läuft auf Port ${PORT}`);
});
