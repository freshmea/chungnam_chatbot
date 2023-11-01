/*
 * The sample smart contract for documentation topic:
 * Managing IOT Data on Hyperledger Blockchain
 */

 class SmartContract {

    // Define the IotData structure, with 2 properties. Structure tags are used by encoding/json library
    class IotData {
        constructor(temperature, humidity) {
            this.temperature = temperature;
            this.humidity = humidity;
        }
    }

    async Init(APIstub) {
        return { status: 200, message: "Instantiated successfully" };
    }

    async Invoke(APIstub) {
        const { fcn, params } = APIstub.getFunctionAndParameters();

        switch (fcn) {
            case "queryIotData":
                return this.queryIotData(APIstub, params);
            case "initLedger":
                return this.initLedger(APIstub);
            case "createIotData":
                return this.createIotData(APIstub, params);
            default:
                return { status: 404, message: "Invalid Smart Contract function name." };
        }
    }

    async queryIotData(APIstub, params) {
        if (params.length !== 1) {
            return { status: 400, message: "Incorrect number of arguments. Expecting 1" };
        }

        const iotDataAsBytes = await APIstub.getState(params[0]);
        return { status: 200, message: iotDataAsBytes.toString() };
    }

    async initLedger(APIstub) {
        const iotDatas = [
            new IotData("0", "0")
        ];

        for (let i = 0; i < iotDatas.length; i++) {
            console.log("i is ", i);
            const iotDataAsBytes = Buffer.from(JSON.stringify(iotDatas[i]));
            await APIstub.putState("IotData" + i, iotDataAsBytes);
            console.log("Added", iotDatas[i]);
        }

        return { status: 200, message: "Ledger initialized successfully" };
    }

    async createIotData(APIstub, params) {
        if (params.length !== 3) {
            return { status: 400, message: "Incorrect number of arguments. Expecting 3" };
        }

        const iotData = new IotData(params[1], params[2]);
        const iotDataAsBytes = Buffer.from(JSON.stringify(iotData));
        await APIstub.putState(params[0], iotDataAsBytes);

        return { status: 200, message: "IotData created successfully" };
    }
}

module.exports = SmartContract;