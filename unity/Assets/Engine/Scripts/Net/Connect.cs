using System.Net.Sockets;
using UnityEngine;
using System.Net;
using System;
using System.Threading;

public class Connect
{
    UdpClient sendClient, recviceClient;
    IPEndPoint point1, point2;
    Thread thread;

    public void Initial(int port1, int port2)
    {
        var remoteIp = IPAddress.Parse("127.0.0.1");
        point1 = new IPEndPoint(remoteIp, port1);
        sendClient = new UdpClient(port1);

        //recv
        recviceClient = new UdpClient(port2);
        point2 = new IPEndPoint(IPAddress.Any, port2);
        thread = new Thread(Receive);
        thread.IsBackground = true;
        thread.Start();
        Debug.Log("initial success send port:" + port1 + "  recv port:" + port2);
    }


    public void Send()
    {
        try
        {
            var data = new byte[] { 0x00, 0x11, 0x22, 0x33 };
            sendClient.Send(data, data.Length, point1);
            Debug.Log("send data");
        }
        catch (Exception ex)
        {
            Debug.LogError("udp send error:" + ex.Message);
        }
    }

    public void Receive()
    {
        while (true)
        {
            try
            {
                byte[] recivcedata = recviceClient.Receive(ref point2);
                string str = "rcv: ";
                for (int i = 0; i < recivcedata.Length; i++)
                {
                    str += recivcedata[i].ToString("x2") + " ";
                }
                Debug.Log(str);
            }
            catch (Exception ex)
            {
                Debug.LogError("udp recv error:" + ex.Message);
                break;
            }
        }
    }


    public void Quit()
    {
        if (sendClient != null)
        {
            sendClient.Close();
        }
        if (recviceClient != null)
        {
            recviceClient.Close();
        }
        if (thread != null)
        {
            thread.Abort();
        }
    }

}