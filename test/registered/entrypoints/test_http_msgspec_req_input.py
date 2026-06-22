import unittest
from typing import Annotated

from fastapi import Body, FastAPI
from fastapi.testclient import TestClient

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import (  # noqa: E402
    ParseFunctionCallReq,
    UpdateWeightFromDiskReqInput,
    VertexGenerateReqInput,
)

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestHttpMsgspecReqInput(CustomTestCase):
    def test_msgspec_reqs_are_fastapi_body_models(self):
        app = FastAPI()

        @app.post("/update_weights_from_disk")
        def update_weights_from_disk(
            obj: Annotated[UpdateWeightFromDiskReqInput, Body()],
        ):
            return {
                "model_path": obj.model_path,
                "manifest": obj.manifest,
                "rid": obj.rid,
                "http_worker_ipc": obj.http_worker_ipc,
            }

        @app.post("/parse_function_call")
        def parse_function_call(obj: Annotated[ParseFunctionCallReq, Body()]):
            return {
                "text": obj.text,
                "tool_name": obj.tools[0].function.name,
                "rid": obj.rid,
                "http_worker_ipc": obj.http_worker_ipc,
            }

        @app.post("/vertex_generate")
        def vertex_generate(obj: Annotated[VertexGenerateReqInput, Body()]):
            return {
                "instances": obj.instances,
                "parameters": obj.parameters,
                "rid": obj.rid,
                "http_worker_ipc": obj.http_worker_ipc,
            }

        openapi = app.openapi()
        for path, schema_name in (
            ("/update_weights_from_disk", "UpdateWeightFromDiskReqInput"),
            ("/parse_function_call", "ParseFunctionCallReq"),
            ("/vertex_generate", "VertexGenerateReqInput"),
        ):
            operation = openapi["paths"][path]["post"]
            self.assertIn("requestBody", operation)
            self.assertNotIn("parameters", operation)
            schema_properties = openapi["components"]["schemas"][schema_name][
                "properties"
            ]
            self.assertIn("rid", schema_properties)
            self.assertIn("http_worker_ipc", schema_properties)

        client = TestClient(app)

        response = client.post(
            "/update_weights_from_disk",
            json={
                "model_path": "/tmp/model",
                "manifest": {"revision": 1},
                "rid": "update-rid",
                "http_worker_ipc": "worker-0",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "model_path": "/tmp/model",
                "manifest": {"revision": 1},
                "rid": "update-rid",
                "http_worker_ipc": "worker-0",
            },
        )

        response = client.post(
            "/parse_function_call",
            json={
                "text": '<tool_call>{"name":"weather","arguments":{}}</tool_call>',
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "weather",
                            "description": "Get weather",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
                "rid": "parse-rid",
                "http_worker_ipc": "worker-1",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["tool_name"], "weather")
        self.assertEqual(response.json()["rid"], "parse-rid")

        response = client.post(
            "/vertex_generate",
            json={
                "instances": [{"prompt": "hello"}],
                "parameters": {"temperature": 0.1},
                "rid": "vertex-rid",
                "http_worker_ipc": "worker-2",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["instances"], [{"prompt": "hello"}])
        self.assertEqual(response.json()["rid"], "vertex-rid")

        self.assertEqual(
            client.post("/update_weights_from_disk", json={"rid": "x"}).status_code,
            422,
        )
        self.assertEqual(
            client.post("/parse_function_call", json={"tools": []}).status_code,
            422,
        )
        self.assertEqual(
            client.post("/vertex_generate", json={"parameters": {}}).status_code,
            422,
        )


if __name__ == "__main__":
    unittest.main()
