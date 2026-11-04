"""Check the Order Ops sample catalog without starting the HTTP server."""
import asyncio
import unittest

import mcp_server


class OrderOpsServerTests(unittest.TestCase):
    def setUp(self):
        mcp_server.reset_catalog()

    def test_catalog_and_shipment_rules(self):
        paid = mcp_server.list_orders("paid")
        self.assertEqual({order["id"] for order in paid}, {"ORD-1001", "ORD-1002"})
        self.assertEqual(mcp_server.get_order("missing"), {"error": "Order not found: missing"})
        self.assertEqual(mcp_server.check_inventory("SKU-MON-27")["on_hand"], 3)
        self.assertIn("error", mcp_server.check_inventory("SKU-NONE"))

        pending = mcp_server.create_shipment("ORD-1003", "UPS")
        self.assertIn("pending", pending["error"])
        self.assertEqual(mcp_server.get_order("ORD-1003")["status"], "pending")

        short = mcp_server.create_shipment("ORD-1002", "FedEx")
        self.assertEqual(short["error"], "Insufficient inventory for SKU-MON-27")
        self.assertEqual(mcp_server.check_inventory("SKU-MON-27")["on_hand"], 3)
        self.assertEqual(mcp_server.get_order("ORD-1002")["status"], "paid")

        shipment = mcp_server.create_shipment("ORD-1001", "UPS")
        self.assertEqual(shipment["shipment_id"], "SHP-0001")
        self.assertEqual(shipment["carrier"], "UPS")
        self.assertEqual(mcp_server.check_inventory("SKU-LAP-14")["on_hand"], 10)
        self.assertEqual(mcp_server.get_order("ORD-1001")["status"], "shipped")
        again = mcp_server.create_shipment("ORD-1001")
        self.assertIn("shipped", again["error"])

    def test_tools_are_registered(self):
        tools = asyncio.run(mcp_server.mcp.list_tools())
        self.assertEqual({tool.name for tool in tools}, {"list_orders", "get_order", "check_inventory", "create_shipment"})


if __name__ == "__main__":
    unittest.main()
