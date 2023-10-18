import streamlit as st
import time


# Clicker program
def main():
    # make Singleton instance
    st.title("클리커 프로그램")
    # print timer
    if "cnt" not in st.session_state:
        st.session_state["cnt"] = 0
        st.session_state["time"] = time.time()
        st.write("init")
    # update automatically

    st.write("Timer: ", 100.0 + st.session_state["time"] - time.time())
    bt = st.button("click button")
    if bt:
        st.session_state["cnt"] += 1
        st.write(st.session_state["cnt"])
    else:
        st.write("No Data")

    checkbox_btn = st.checkbox("Checktbox Button")

    if checkbox_btn:
        st.write("Great!")
        st.write(st.session_state["cnt"])
    print("run")


if __name__ == "__main__":
    main()
