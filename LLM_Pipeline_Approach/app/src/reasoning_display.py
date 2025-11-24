"""
UI components for displaying LLM reasoning to users.
"""
import streamlit as st


def show_reasoning(key, title=None, expanded=False):
    """
    Display LLM reasoning from session state if available.
    
    Args:
        key: Key in st.session_state.llm_reasoning
        title: Custom title for the expander
        expanded: Whether expander should be open by default
    """
    if 'llm_reasoning' not in st.session_state:
        return
    
    if key not in st.session_state.llm_reasoning:
        return
    
    reasoning_data = st.session_state.llm_reasoning[key]
    
    # Map keys to icons
    icon_map = {
        'target_selection': '🎯',
        'null_filling': '🔧',
        'encoding': '🔢',
        'test_ratio': '📊',
        'balance_strategy': '⚖️',
        'model_selection': '🤖',
        'cluster_model_selection': '🔍',
        'regression_model_selection': '📈'
    }
    
    icon = icon_map.get(key, '🧠')
    display_title = title or key.replace('_', ' ').title()
    
    with st.expander(f"{icon} **AI Reasoning:** {display_title}", expanded=expanded):
        if isinstance(reasoning_data, dict):
            # Handle model selection with overall reasoning
            if 'reasoning' in reasoning_data and 'overall' in reasoning_data:
                if reasoning_data['overall']:
                    st.info(f"**💡 Overall Strategy**\n\n{reasoning_data['overall']}")
                    st.divider()
                
                for item, reason in reasoning_data['reasoning'].items():
                    st.markdown(f"**✓ {item}**")
                    st.markdown(reason)
                    st.markdown("---")
            else:
                # Handle simple dict of reasoning
                for item, reason in reasoning_data.items():
                    st.markdown(f"**✓ {item}**")
                    st.markdown(reason)
                    st.markdown("---")
        else:
            # Handle simple string reasoning
            st.markdown(f"**💡 Reasoning**\n\n{reasoning_data}")


def show_all_reasoning_summary():
    """
    Display a summary panel of all AI decisions and reasoning.
    Call this at the end of the pipeline.
    """
    if 'llm_reasoning' not in st.session_state:
        return
    
    if not st.session_state.llm_reasoning:
        return
    
    st.divider()
    st.subheader("📋 Complete AI Decision Summary")
    
    with st.expander("🔍 View All AI Decisions and Reasoning", expanded=False):
        # Create tabs for each decision
        decision_keys = list(st.session_state.llm_reasoning.keys())
        
        if decision_keys:
            tabs = st.tabs([key.replace('_', ' ').title() for key in decision_keys])
            
            for idx, key in enumerate(decision_keys):
                with tabs[idx]:
                    reasoning_data = st.session_state.llm_reasoning[key]
                    
                    if isinstance(reasoning_data, dict):
                        if 'reasoning' in reasoning_data and 'overall' in reasoning_data:
                            if reasoning_data['overall']:
                                st.info(f"**Overall Strategy:**\n\n{reasoning_data['overall']}")
                                st.divider()
                            
                            for item, reason in reasoning_data['reasoning'].items():
                                st.markdown(f"**{item}:**")
                                st.markdown(reason)
                                st.divider()
                        else:
                            for item, reason in reasoning_data.items():
                                st.markdown(f"**{item}:**")
                                st.markdown(reason)
                                st.divider()
                    else:
                        st.markdown(reasoning_data)


def clear_reasoning():
    """Clear all stored reasoning from session state."""
    if 'llm_reasoning' in st.session_state:
        st.session_state.llm_reasoning = {}

