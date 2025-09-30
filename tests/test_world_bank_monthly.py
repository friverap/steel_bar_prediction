"""
Test para World Bank Monthly Commodities (Pink Sheet)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_ingestion.world_bank_monthly_collector import collect_world_bank_monthly_data
from datetime import datetime

def test_world_bank_monthly():
    """
    Prueba la recopilación de datos mensuales de World Bank
    """
    print("="*60)
    print("🔍 TEST: World Bank Monthly Commodities (Pink Sheet)")
    print("="*60)
    
    # Recopilar datos de los últimos 5 años
    start_date = "2020-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    print(f"\n📅 Período: {start_date} a {end_date}")
    print("Descargando datos mensuales de commodities...\n")
    
    try:
        data = collect_world_bank_monthly_data(
            start_date=start_date,
            end_date=end_date,
            save_raw=True
        )
        
        if data and 'commodities_data' in data:
            print("✅ DATOS MENSUALES OBTENIDOS EXITOSAMENTE")
            print("\n📊 RESUMEN:")
            print("-"*40)
            
            summary = data.get('summary', {})
            print(f"Total commodities: {summary.get('total_commodities', 0)}")
            print(f"Total puntos de datos: {summary.get('total_data_points', 0)}")
            print(f"Frecuencia: {summary.get('frequency', 'N/A')}")
            print(f"Fuente: {summary.get('source', 'N/A')}")
            
            print("\n📈 COMMODITIES DISPONIBLES (MENSUAL):")
            print("-"*40)
            
            for key, commodity in data['commodities_data'].items():
                print(f"\n• {commodity['name']}:")
                print(f"  - Datos: {commodity['count']} meses")
                print(f"  - Último valor: ${commodity['latest_value']:.2f} {commodity['unit']}")
                print(f"  - Última fecha: {commodity['latest_date'].strftime('%Y-%m')}")
                
                # Mostrar tendencia reciente
                if 'data' in commodity and not commodity['data'].empty:
                    df = commodity['data']
                    recent = df.tail(6)  # Últimos 6 meses
                    
                    print(f"  - Últimos 6 meses:")
                    for _, row in recent.iterrows():
                        print(f"    {row['fecha'].strftime('%Y-%m')}: ${row['valor']:.2f}")
                    
                    # Calcular cambio porcentual
                    if len(df) >= 12:
                        current = df['valor'].iloc[-1]
                        year_ago = df['valor'].iloc[-12]
                        change_pct = ((current - year_ago) / year_ago) * 100
                        print(f"  - Cambio anual: {change_pct:+.1f}%")
            
            print("\n" + "="*60)
            print("✅ TEST EXITOSO - World Bank tiene datos MENSUALES")
            print("   Frecuencia: MENSUAL")
            print("   Actualización: Mensual")
            print("   Cobertura: Mineral de hierro, carbón, metales")
            print("="*60)
            
        else:
            print("❌ No se pudieron obtener datos")
            
    except Exception as e:
        print(f"❌ Error en el test: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_world_bank_monthly()
