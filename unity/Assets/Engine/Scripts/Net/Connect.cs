using System.Net.Sockets;
using UnityEngine;
using System.Net;
using System;
using System.Threading;
using System.Text;

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


    public void Send(string msg)
    {
        try
        {
            var data = Encoding.ASCII.GetBytes(msg);
            sendClient.Send(data, data.Length, point1);
            Debug.Log("send " + msg);
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
            byte[] recivcedata = recviceClient.Receive(ref point2);
            string str = Encoding.ASCII.GetString(recivcedata, 0, recivcedata.Length);
            Debug.Log("rcv: " + str);
            if (str.StartsWith("rcv"))
            {
                str = str.Substring(3);
                Send("recv msg by client");
            }
            if (str.Equals("quit"))
            {
                break;
            }
        }
        Quit();
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