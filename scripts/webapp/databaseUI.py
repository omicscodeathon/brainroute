import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from database import get_database

def show_database_page():
    """
    Display the database browser page with molecule predictions.
    """
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: 700;
            color: #1f77b4;
            margin-bottom: 1rem;
            text-align: center;
        }
        .stat-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .stat-number {
            font-size: 2.5rem;
            font-weight: bold;
            margin: 0;
        }
        .stat-label {
            font-size: 0.9rem;
            opacity: 0.9;
            margin-top: 0.5rem;
        }
        .filter-section {
            background-color: #f8f9fa;
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        .molecule-card {
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem 0;
            background-color: white;
        }
        .bbb-permeable {
            color: #28a745;
            font-weight: bold;
        }
        .bbb-non-permeable {
            color: #dc3545;
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="main-header">🧬 Molecule Database Browser</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Get database instance
    db = get_database()
    
    # Display statistics
    show_statistics(db)
    
    st.markdown("---")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Browse All", 
        "🔍 Search & Filter", 
        "📈 Analytics", 
        "💾 Export Data"
    ])
    
    with tab1:
        show_all_molecules(db)
    
    with tab2:
        show_search_filter(db)
    
    with tab3:
        show_analytics(db)
    
    with tab4:
        show_export_options(db)


def show_statistics(db):
    """Display key statistics in cards."""
    stats = db.get_statistics()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
            <div class="stat-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
                <div class="stat-number">{stats['total_predictions']}</div>
                <div class="stat-label">Total Predictions</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="stat-card" style="background: linear-gradient(135deg, #28a745 0%, #20c997 100%);">
                <div class="stat-number">{stats['bbb_permeable']}</div>
                <div class="stat-label">BBB Permeable</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class="stat-card" style="background: linear-gradient(135deg, #dc3545 0%, #e83e8c 100%);">
                <div class="stat-number">{stats['bbb_non_permeable']}</div>
                <div class="stat-label">BBB Non-Permeable</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_mw = stats['avg_molecular_weight'] if stats['avg_molecular_weight'] else 0
        st.markdown(f"""
            <div class="stat-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                <div class="stat-number">{avg_mw:.1f}</div>
                <div class="stat-label">Avg. Mol. Weight</div>
            </div>
        """, unsafe_allow_html=True)


def show_all_molecules(db):
    """Display all molecules in a table."""
    st.subheader("📋 All Predictions")
    
    # Pagination controls
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        page_size = st.selectbox("Rows per page", [10, 25, 50, 100], index=1)
    
    df = db.get_all_predictions()
    
    if df.empty:
        st.info("🔍 No predictions in the database yet. Start predicting molecules to build your database!")
        return
    
    st.write(f"**Total Records:** {len(df)}")
    
    # Format the dataframe for display
    display_df = df.copy()
    display_df['prediction'] = display_df['prediction'].map({
        1: '✅ Permeable', 
        0: '❌ Non-Permeable'
    })
    
    # Round numeric columns
    numeric_cols = ['prediction_probability', 'molecular_weight', 'logp', 'tpsa']
    for col in numeric_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].round(3)
    
    # Format date
    if 'prediction_date' in display_df.columns:
        display_df['prediction_date'] = pd.to_datetime(display_df['prediction_date']).dt.strftime('%Y-%m-%d %H:%M')
    
    # Select columns to display
    display_cols = [
        'id', 'molecule_name', 'smiles', 'chembl_id', 'prediction', 
        'prediction_probability', 'molecular_weight', 'logp', 
        'h_bond_donors', 'h_bond_acceptors', 'prediction_date'
    ]
    
    display_cols = [col for col in display_cols if col in display_df.columns]
    
    # Paginate
    total_pages = (len(display_df) - 1) // page_size + 1
    page = st.number_input("Page", min_value=1, max_value=max(1, total_pages), value=1)
    
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    
    st.dataframe(
        display_df[display_cols].iloc[start_idx:end_idx],
        use_container_width=True,
        hide_index=True
    )
    
    st.caption(f"Showing {start_idx + 1}-{min(end_idx, len(df))} of {len(df)} records")


def show_search_filter(db):
    """Show search and filter interface."""
    st.subheader("🔍 Search & Filter")
    
    col1, col2 = st.columns(2)
    
    with col1:
        search_query = st.text_input(
            "🔎 Search",
            placeholder="Enter SMILES, molecule name, or ChEMBL ID...",
            help="Search across SMILES, molecule names, and ChEMBL IDs"
        )
    
    with col2:
        bbb_filter = st.selectbox(
            "BBB Permeability",
            ["All", "Permeable Only", "Non-Permeable Only"]
        )
    
    col3, col4 = st.columns(2)
    
    with col3:
        min_mw = st.number_input("Min Molecular Weight", min_value=0.0, value=0.0, step=10.0)
    
    with col4:
        max_mw = st.number_input("Max Molecular Weight", min_value=0.0, value=1000.0, step=10.0)
    
    # Apply filters
    bbb_permeable = None
    if bbb_filter == "Permeable Only":
        bbb_permeable = True
    elif bbb_filter == "Non-Permeable Only":
        bbb_permeable = False
    
    if st.button("🔍 Apply Filters", type="primary"):
        results_df = db.search_molecules(
            query=search_query if search_query else None,
            bbb_permeable=bbb_permeable,
            min_mw=min_mw if min_mw > 0 else None,
            max_mw=max_mw if max_mw < 1000 else None
        )
        
        if results_df.empty:
            st.warning("No molecules found matching your criteria.")
        else:
            st.success(f"Found {len(results_df)} molecule(s)")
            
            # Format display
            results_df['prediction'] = results_df['prediction'].map({
                1: '✅ Permeable', 
                0: '❌ Non-Permeable'
            })
            
            display_cols = [
                'id', 'molecule_name', 'smiles', 'chembl_id', 'prediction', 
                'prediction_probability', 'molecular_weight', 'prediction_date'
            ]
            display_cols = [col for col in display_cols if col in results_df.columns]
            
            st.dataframe(results_df[display_cols], use_container_width=True, hide_index=True)


def show_analytics(db):
    """Show analytics and visualizations."""
    st.subheader("📈 Database Analytics")
    
    df = db.get_all_predictions()
    
    if df.empty:
        st.info("No data available for analytics yet.")
        return
    
    # BBB Permeability Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### BBB Permeability Distribution")
        perm_counts = df['prediction'].value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=['Permeable', 'Non-Permeable'],
            values=[perm_counts.get(1, 0), perm_counts.get(0, 0)],
            hole=0.4,
            marker=dict(colors=['#28a745', '#dc3545'])
        )])
        
        fig.update_layout(
            showlegend=True,
            height=300,
            margin=dict(t=0, b=0, l=0, r=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Predictions Over Time")
        df['prediction_date'] = pd.to_datetime(df['prediction_date'])
        daily_counts = df.groupby(df['prediction_date'].dt.date).size().reset_index()
        daily_counts.columns = ['date', 'count']
        
        fig = px.line(
            daily_counts,
            x='date',
            y='count',
            markers=True,
            labels={'date': 'Date', 'count': 'Number of Predictions'}
        )
        
        fig.update_layout(
            height=300,
            margin=dict(t=20, b=0, l=0, r=0),
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Molecular Weight Distribution
    st.markdown("### Molecular Weight Distribution")
    
    if 'molecular_weight' in df.columns:
        df_mw = df[df['molecular_weight'].notna()]
        
        if not df_mw.empty:
            fig = px.histogram(
                df_mw,
                x='molecular_weight',
                color='prediction',
                barmode='overlay',
                labels={'molecular_weight': 'Molecular Weight', 'prediction': 'BBB Permeable'},
                color_discrete_map={0: '#dc3545', 1: '#28a745'}
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Property Correlation
    st.markdown("### Molecular Property Correlations")
    
    numeric_cols = ['molecular_weight', 'logp', 'h_bond_donors', 'h_bond_acceptors', 'tpsa', 'rotatable_bonds']
    available_cols = [col for col in numeric_cols if col in df.columns and df[col].notna().any()]
    
    if len(available_cols) >= 2:
        col1, col2 = st.columns(2)
        
        with col1:
            x_axis = st.selectbox("X-axis", available_cols, index=0)
        with col2:
            y_axis = st.selectbox("Y-axis", available_cols, index=1 if len(available_cols) > 1 else 0)
        
        df_plot = df[[x_axis, y_axis, 'prediction']].dropna()
        
        if not df_plot.empty:
            fig = px.scatter(
                df_plot,
                x=x_axis,
                y=y_axis,
                color='prediction',
                color_discrete_map={0: '#dc3545', 1: '#28a745'},
                labels={'prediction': 'BBB Permeable'}
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)


def show_export_options(db):
    """Show export options."""
    st.subheader("💾 Export Database")
    
    df = db.get_all_predictions()
    
    if df.empty:
        st.info("No data to export yet.")
        return
    
    st.write(f"**Total records available for export:** {len(df)}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📥 Download as CSV")
        st.write("Export all prediction data to a CSV file.")
        
        csv = df.to_csv(index=False).encode('utf-8')
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"bbb_predictions_{timestamp}.csv"
        
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=filename,
            mime='text/csv',
            type="primary"
        )
    
    with col2:
        st.markdown("### 📊 Download as Excel")
        st.write("Export all prediction data to an Excel file.")
        
        # Convert to Excel
        from io import BytesIO
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Predictions')
        
        excel_data = output.getvalue()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"bbb_predictions_{timestamp}.xlsx"
        
        st.download_button(
            label="📥 Download Excel",
            data=excel_data,
            file_name=filename,
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            type="primary"
        )
    
    st.markdown("---")
    
    # Export summary statistics
    st.markdown("### 📈 Export Summary Report")
    
    stats = db.get_statistics()
    
    summary_text = f"""
# BBB Permeability Prediction Database Summary
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Statistics
- Total Predictions: {stats['total_predictions']}
- BBB Permeable: {stats['bbb_permeable']}
- BBB Non-Permeable: {stats['bbb_non_permeable']}
- Average Molecular Weight: {stats['avg_molecular_weight']:.2f if stats['avg_molecular_weight'] else 'N/A'}
- First Prediction: {stats['first_prediction']}
- Latest Prediction: {stats['latest_prediction']}

## Database Coverage
This database contains predictions for {stats['total_predictions']} unique molecules, 
helping researchers understand BBB permeability patterns.
    """
    
    st.download_button(
        label="📥 Download Summary Report",
        data=summary_text,
        file_name=f"bbb_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime='text/plain'
    )