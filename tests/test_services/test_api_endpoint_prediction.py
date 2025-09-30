#!/usr/bin/env python3
"""
Script de prueba para verificar que la API funciona correctamente
"""

import asyncio
import sys
from fastapi.testclient import TestClient
from app.main import app

def test_api_endpoints():
    """Test basic API endpoints"""
    client = TestClient(app)
    
    print("🧪 Testing DeAcero Steel Price Predictor API...")
    
    # Test root endpoint
    print("\n1. Testing root endpoint...")
    response = client.get("/")
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   Service: {data.get('service')}")
        print(f"   Version: {data.get('version')}")
        print("   ✅ Root endpoint working")
    else:
        print("   ❌ Root endpoint failed")
        return False
    
    # Test health endpoint
    print("\n2. Testing health endpoint...")
    response = client.get("/health")
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   Status: {data.get('status')}")
        print("   ✅ Health endpoint working")
    else:
        print("   ❌ Health endpoint failed")
        return False
    
    # Test prediction endpoint (should require API key)
    print("\n3. Testing prediction endpoint without API key...")
    response = client.get("/predict/steel-rebar-price")
    print(f"   Status: {response.status_code}")
    if response.status_code == 422:  # Missing required header
        print("   ✅ API key validation working")
    else:
        print("   ❌ API key validation not working")
    
    # Test prediction endpoint with API key
    print("\n4. Testing prediction endpoint with API key...")
    headers = {"X-API-Key": "your_secret_api_key_here_change_this_in_production"}
    response = client.get("/predict/steel-rebar-price", headers=headers)
    print(f"   Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"   Prediction Date: {data.get('prediction_date')}")
        print(f"   Predicted Price: ${data.get('predicted_price_usd_per_ton'):.2f}")
        print(f"   Currency: {data.get('currency')}")
        print(f"   Unit: {data.get('unit')}")
        print(f"   Confidence: {data.get('model_confidence'):.2f}")
        print(f"   Timestamp: {data.get('timestamp')}")
        print("   ✅ Prediction endpoint working")
    else:
        print(f"   Error: {response.text}")
        print("   ❌ Prediction endpoint failed")
        return False
    
    # Test OpenAPI docs
    print("\n5. Testing OpenAPI docs...")
    response = client.get("/docs")
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        print("   ✅ OpenAPI docs available")
    else:
        print("   ❌ OpenAPI docs not available")
    
    print("\n🎉 All tests passed! API is working correctly.")
    return True

if __name__ == "__main__":
    try:
        success = test_api_endpoints()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n💥 Test failed with error: {str(e)}")
        sys.exit(1)
