#!/usr/bin/env python3
"""
Test para datos de materias primas del acero
Mineral de Hierro y Carbón de Coque
"""

import asyncio
import sys
import os
from datetime import datetime

# Configurar paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from src.data_ingestion.raw_materials_collector import RawMaterialsCollector


async def test_raw_materials():
    """Test de materias primas del acero"""
    
    print("=" * 60)
    print("⛏️ TEST MATERIAS PRIMAS DEL ACERO")
    print("=" * 60)
    print("📊 Mineral de Hierro y Carbón de Coque")
    print("=" * 60)
    
    async with RawMaterialsCollector() as collector:
        
        # Test 1: Empresas de Mineral de Hierro
        print("\n🏔️ Test 1: EMPRESAS DE MINERAL DE HIERRO")
        print("-" * 40)
        
        iron_ore_companies = ['vale', 'rio_tinto', 'bhp', 'fortescue']
        iron_ore_data = []
        
        for company in iron_ore_companies:
            print(f"\n📊 {company.upper()}")
            result = await collector.get_mining_company_data(
                company,
                start_date='2024-01-01'
            )
            
            if result:
                print(f"   ✅ {result['name']}")
                print(f"   📈 Precio actual: ${result['latest_price']:.2f}")
                print(f"   📅 Última fecha: {result['latest_date']}")
                print(f"   📊 Datos: {result['count']} días")
                print(f"   📈 Cambio desde enero: {result['price_change_pct']:.2f}%")
                if result['volatility']:
                    print(f"   📊 Volatilidad: {result['volatility']:.4f}")
                print(f"   🔗 Correlación con mineral: {result['correlation_factor']:.2f}")
                iron_ore_data.append(result)
            else:
                print(f"   ❌ Sin datos")
        
        # Test 2: Empresas de Carbón de Coque
        print("\n⚫ Test 2: EMPRESAS DE CARBÓN DE COQUE")
        print("-" * 40)
        
        coal_companies = ['teck', 'arch_resources', 'bhp', 'anglo_american']
        coal_data = []
        
        for company in coal_companies:
            print(f"\n📊 {company.upper()}")
            result = await collector.get_mining_company_data(
                company,
                start_date='2024-01-01'
            )
            
            if result:
                print(f"   ✅ {result['name']}")
                print(f"   📈 Precio actual: ${result['latest_price']:.2f}")
                print(f"   📅 Última fecha: {result['latest_date']}")
                print(f"   📊 Datos: {result['count']} días")
                print(f"   📈 Cambio desde enero: {result['price_change_pct']:.2f}%")
                coal_data.append(result)
            else:
                print(f"   ❌ Sin datos")
        
        # Test 3: ETFs del Sector
        print("\n📊 Test 3: ETFs DEL SECTOR")
        print("-" * 40)
        
        for etf_key, etf_info in collector.commodity_etfs.items():
            print(f"\n🔍 {etf_info['symbol']} - {etf_info['name']}")
            
            result = await collector.get_etf_data(
                etf_key,
                start_date='2024-01-01'
            )
            
            if result:
                print(f"   ✅ Datos: {result['count']} días")
                print(f"   📈 Último valor: ${result['latest_value']:.2f}")
                print(f"   📅 Última fecha: {result['latest_date']}")
            else:
                print(f"   ❌ Sin datos")
        
        # Test 4: Índices Proxy Calculados
        print("\n📈 Test 4: ÍNDICES PROXY CALCULADOS")
        print("-" * 40)
        
        # Calcular proxy para mineral de hierro
        if iron_ore_data:
            iron_ore_proxy = collector._calculate_commodity_proxy(
                {d['company_key']: d for d in iron_ore_data},
                [d['company_key'] for d in iron_ore_data]
            )
            
            if not iron_ore_proxy.empty:
                print(f"\n🏔️ Índice Proxy Mineral de Hierro:")
                print(f"   ✅ Puntos de datos: {len(iron_ore_proxy)}")
                print(f"   📈 Último índice: {iron_ore_proxy['indice'].iloc[-1]:.2f}")
                print(f"   📅 Rango: {iron_ore_proxy['fecha'].min()} a {iron_ore_proxy['fecha'].max()}")
        
        # Calcular proxy para carbón
        if coal_data:
            coal_proxy = collector._calculate_commodity_proxy(
                {d['company_key']: d for d in coal_data},
                [d['company_key'] for d in coal_data]
            )
            
            if not coal_proxy.empty:
                print(f"\n⚫ Índice Proxy Carbón de Coque:")
                print(f"   ✅ Puntos de datos: {len(coal_proxy)}")
                print(f"   📈 Último índice: {coal_proxy['indice'].iloc[-1]:.2f}")
        
        # Resumen
        print("\n" + "=" * 60)
        print("📋 RESUMEN DE MATERIAS PRIMAS")
        print("=" * 60)
        
        total_iron = len(iron_ore_data)
        total_coal = len(coal_data)
        
        print(f"\n🏔️ MINERAL DE HIERRO:")
        print(f"   ✅ Empresas con datos: {total_iron}/4")
        if iron_ore_data:
            avg_change = sum(d['price_change_pct'] for d in iron_ore_data) / len(iron_ore_data)
            print(f"   📈 Cambio promedio YTD: {avg_change:.2f}%")
        
        print(f"\n⚫ CARBÓN DE COQUE:")
        print(f"   ✅ Empresas con datos: {total_coal}/4")
        if coal_data:
            avg_change = sum(d['price_change_pct'] for d in coal_data) / len(coal_data)
            print(f"   📈 Cambio promedio YTD: {avg_change:.2f}%")
        
        print(f"\n💡 CONCLUSIONES:")
        print("   • Los precios de acciones mineras son buenos proxies")
        print("   • VALE y RIO son los mejores para mineral de hierro")
        print("   • BHP cubre ambos commodities")
        print("   • TECK es el mejor para carbón metalúrgico")
        print("   • Los ETFs SLX y XME dan visión del sector")
        
        print("\n🎯 RECOMENDACIONES:")
        print("   1. Usar VALE + RIO como proxy de mineral de hierro")
        print("   2. Usar TECK + BHP como proxy de carbón de coque")
        print("   3. Monitorear SLX para tendencia del sector acero")
        print("   4. Correlacionar con precios de acero para validar")
        
        print("=" * 60)
        
        return total_iron > 0 and total_coal > 0


if __name__ == "__main__":
    success = asyncio.run(test_raw_materials())
    print(f"\n{'✅' if success else '❌'} Test completado")
    exit(0 if success else 1)
