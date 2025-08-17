# Railway Deployment Guide

## Environment Variables Required

Add these in Railway's Variables section:

1. **ROBOFLOW_API_KEY** (Required for CubiCasa detection)
   ```
   ROBOFLOW_API_KEY=ZdkZuHA7GF5NdOfCwQo8
   ```

2. **RAILWAY_ENVIRONMENT** (Auto-set by Railway)
   - This triggers headless mode and /tmp directory usage

3. **PORT** (Auto-set by Railway)
   - Railway will automatically assign a port

## Deployment URL

After deployment, generate a public domain:
1. Go to Settings → Networking
2. Click "Generate Domain"
3. Your app will be available at the generated URL

## Potential Issues and Solutions

### 1. libGL.so.1 Error
- **Fixed**: Added system packages in nixpacks.toml

### 2. Tesseract OCR Error
- **Fixed**: Added tesseract-ocr to aptPkgs

### 3. Directory Permissions
- **Fixed**: Falls back to /tmp on Railway

### 4. Memory Limits
- If processing large PDFs fails, Railway's free tier has 512MB RAM limit
- Consider upgrading to a paid plan for larger files

### 5. File Upload Size
- Current limit: 100MB (configurable in app.py)

## Testing the Deployment

1. Visit your Railway URL
2. Upload a floor plan image (PNG/JPG)
3. The app will:
   - Detect architectural elements (walls, doors, windows)
   - Show color-coded visualization
   - Provide element counts

## Monitoring

Check Railway logs for any errors:
```bash
railway logs
```

Or view in Railway dashboard → Deployments → View Logs
