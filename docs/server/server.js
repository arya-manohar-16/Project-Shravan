const express = require('express');
const mongoose = require('mongoose');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const dotenv = require('dotenv');
const path = require('path');

dotenv.config();

const app = express();
app.use(express.json());

// Serve static files from the docs/ directory (one level up)
app.use(express.static(path.join(__dirname, '..')));

// MongoDB Connection
mongoose.connect(process.env.MONGODB_URI)
  .then(() => console.log('MongoDB Connected'))
  .catch(err => console.log('MongoDB error:', err));

// Schemas
const userSchema = new mongoose.Schema({
  email: { type: String, required: true, unique: true },
  password: { type: String, required: true }
});

const deviceSchema = new mongoose.Schema({
  code: { type: String, required: true, unique: true },
  owner: { type: mongoose.Schema.Types.ObjectId, ref: 'User', required: true }
});

const User = mongoose.model('User', userSchema);
const Device = mongoose.model('Device', deviceSchema);

// Middleware to verify JWT
const auth = (req, res, next) => {
  const token = req.header('Authorization')?.replace('Bearer ', '');
  if (!token) return res.status(401).json({ error: 'no_token' });
  
  try {
    const verified = jwt.verify(token, process.env.JWT_SECRET);
    req.user = verified;
    next();
  } catch (err) {
    res.status(400).json({ error: 'invalid_token' });
  }
};

// --- ROUTES ---

// 1. Register
app.post('/api/register', async (req, res) => {
  try {
    const salt = await bcrypt.genSalt(10);
    const hashedPassword = await bcrypt.hash(req.body.password, salt);
    
    const newUser = new User({ email: req.body.email, password: hashedPassword });
    await newUser.save();
    res.status(201).json({ message: 'user_created' });
  } catch (err) {
    res.status(400).json({ error: 'registration_failed' });
  }
});

// 2. Login
app.post('/api/login', async (req, res) => {
  try {
    const user = await User.findOne({ email: req.body.email });
    if (!user) return res.status(400).json({ error: 'wrong_credentials' });

    const validPass = await bcrypt.compare(req.body.password, user.password);
    if (!validPass) return res.status(400).json({ error: 'wrong_credentials' });

    const token = jwt.sign({ _id: user._id }, process.env.JWT_SECRET, { expiresIn: '7d' });
    res.json({ token });
  } catch (err) {
    res.status(500).json({ error: 'server_error' });
  }
});

// 3. Get Linked Chair
app.get('/api/chair', auth, async (req, res) => {
  try {
    const device = await Device.findOne({ owner: req.user._id });
    if (!device) return res.status(404).json({ error: 'no_chair_linked' });
    res.json({ code: device.code });
  } catch (err) {
    res.status(500).json({ error: 'server_error' });
  }
});

// 4. Link Chair
app.post('/api/chair', auth, async (req, res) => {
  try {
    const existingDevice = await Device.findOne({ code: req.body.code });
    if (existingDevice) return res.status(400).json({ error: 'code_in_use' });

    const newDevice = new Device({ code: req.body.code, owner: req.user._id });
    await newDevice.save();
    res.json({ message: 'chair_linked', code: newDevice.code });
  } catch (err) {
    res.status(500).json({ error: 'server_error' });
  }
});

// 5. Unlink Chair
app.delete('/api/chair', auth, async (req, res) => {
  try {
    await Device.findOneAndDelete({ owner: req.user._id });
    res.json({ message: 'chair_unlinked' });
  } catch (err) {
    res.status(500).json({ error: 'server_error' });
  }
});

// Serve frontend for any other routes (SPA fallback)
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, '..', 'index.html'));
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));