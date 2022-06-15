if __name__ == "__main__":
    import uvicorn

    uvicorn.run("special_mission.main:app", host="0.0.0.0", port=30001, reload=True)
